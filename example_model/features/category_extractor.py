"""Generic Oscar/BAFTA/SAG category feature extraction pipeline.

Supports both film-level and nominee-level categories for any award system
registered in _SPECS_BY_SYSTEM.  Target category matching and cross-system
precursor feature attachment are driven entirely by msh:realizationOf links
in data/instances/category_alignments.ttl — no ad-hoc string matchers.

Adding support for a new award system requires:
  1. Adding AwardConcept hub nodes and msh:realizationOf links in
     data/instances/category_alignments.ttl
  2. Adding an entry to _SYSTEM_KEYS and _SPECS_BY_SYSTEM below
  3. No Python code changes needed in extract()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import Graph, URIRef

from foundation.loader import load_foundation_graph
from foundation.manifest import PROJECT_ROOT
from .utils import normalize_person_name

from .canonical import (
    MSH,
    _query_nominations,
    extract_canonical_features,
    extract_category_features,
    extract_precursor_features,
)
from .graph_enrichment import extract_graph_enrichment


_REQUIRED_COMMON_COLS = ["year_film", "winner", "film_uri", "film", "nom"]
_REQUIRED_NOMINEE_COLS = ["candidate_id", "candidate_name", "nominee"]

_FILM_CAT_COLS = [
    "acting_noms",
    "directing_nom",
    "writing_noms",
    "cinematography_nom",
    "editing_nom",
    "music_noms",
    "technical_noms",
    "above_line_noms",
    "other_noms",
]
_NOMINEE_HISTORY_COLS = [
    "nominee_previous_noms_sum",
    "nominee_previous_noms_max",
    "nominee_previous_noms_avg",
    "nominee_previous_wins_sum",
    "nominee_previous_wins_max",
    "nominee_previous_wins_avg",
]

# Film-level precursor columns added by extract_precursor_features.
# When the target system owns one of these, exclude it to prevent leakage.
_ALL_FILM_PRECURSOR_COLS = ["pga_winner", "sag_winner", "globe_winner", "dga_winner", "bafta_winner"]

# Specific best-film precursor columns (ensemble cast only).
_BEST_FILM_PRECURSOR_COLS = ["globe_drama_winner", "globe_comedy_winner", "bafta_best_film_winner"]
_FILM_PRECURSOR_EXCL: dict[str, str] = {
    "bafta": "bafta_winner",
    "sag":   "sag_winner",
}

# Maps system alias → exact msh:shortName stored in the graph.
_SYSTEM_KEYS: dict[str, str] = {
    "oscars":        "Oscars",
    "bafta":         "BAFTA",
    "sag":           "SAG",
    "golden_globes": "GOLDEN_GLOBES",
    "dga":           "DGA",
    "pga":           "PGA",
}

MSH_BASE = "http://example.org/ontologies/MovieSHACL3#"


def _concept(local: str) -> str:
    return f"{MSH_BASE}{local}"


# ---------------------------------------------------------------------------
# CategorySpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CategorySpec:
    alias: str
    row_type: str       # "film" or "nominee"
    concept_uri: str    # full msh:AwardConcept URI
    col_prefix: str | None = None   # prefix for history + precursor columns


def _S(alias: str, row_type: str, concept_local: str, col_prefix: str | None = None) -> CategorySpec:
    return CategorySpec(alias=alias, row_type=row_type,
                        concept_uri=_concept(concept_local), col_prefix=col_prefix)


# SAG acting concept URIs used to identify cast members for history aggregation.
# Includes ensemble cast so that prior ensemble nominations/wins (e.g. MBJ's
# Black Panther win) count toward cast_star_power and cast history features.
# Defined here because _concept() must be in scope.
_SAG_ACTING_CONCEPTS = {
    _concept("Concept_BestMaleLeadActor"),
    _concept("Concept_BestFemaleLeadActor"),
    _concept("Concept_BestMaleSupportingActor"),
    _concept("Concept_BestFemaleSupportingActor"),
    _concept("Concept_BestEnsembleCast"),
}


# ---------------------------------------------------------------------------
# Category specs per award system
# ---------------------------------------------------------------------------

_OSCAR_CATEGORY_SPECS: dict[str, CategorySpec] = {
    "best_picture":            _S("best_picture",            "film",    "Concept_BestPicture"),
    "best_actor":              _S("best_actor",              "nominee", "Concept_BestMaleLeadActor",        "actor"),
    "best_actress":            _S("best_actress",            "nominee", "Concept_BestFemaleLeadActor",      "actress"),
    "best_supporting_actor":   _S("best_supporting_actor",   "nominee", "Concept_BestMaleSupportingActor",  "supporting_actor"),
    "best_supporting_actress": _S("best_supporting_actress", "nominee", "Concept_BestFemaleSupportingActor","supporting_actress"),
    "best_director":           _S("best_director",           "nominee", "Concept_BestDirector",             "director"),
    "best_original_screenplay":_S("best_original_screenplay","film",    "Concept_BestOriginalScreenplay","original_screenplay"),
    "best_adapted_screenplay": _S("best_adapted_screenplay", "film",    "Concept_BestAdaptedScreenplay", "adapted_screenplay"),
    "cinematography":          _S("cinematography",          "nominee", "Concept_BestCinematography",       "cinematography"),
    "film_editing":            _S("film_editing",            "film",    "Concept_BestFilmEditing",           "film_editing"),
    "original_score":          _S("original_score",          "nominee", "Concept_BestOriginalScore",        "original_score"),
    "original_song":           _S("original_song",           "film",    "Concept_BestOriginalSong",          "original_song"),
    "production_design":       _S("production_design",       "film",    "Concept_BestProductionDesign",      "production_design"),
    "costume_design":          _S("costume_design",          "nominee", "Concept_BestCostumeDesign",        "costume_design"),
    "sound":                   _S("sound",                   "film",    "Concept_BestSound",                 "sound"),
    "visual_effects":          _S("visual_effects",          "film",    "Concept_BestVisualEffects",         "visual_effects"),
    "makeup":                  _S("makeup",                  "film",    "Concept_BestMakeup",                "makeup"),
    "international_film":      _S("international_film",      "film",    "Concept_BestInternationalFilm",     "international_film"),
    "documentary":             _S("documentary",             "film",    "Concept_BestDocumentaryFeature",    "documentary"),
    "animated_film":           _S("animated_film",           "film",    "Concept_BestAnimatedFeature",       "animated_film"),
}

_BAFTA_CATEGORY_SPECS: dict[str, CategorySpec] = {
    "best_film":                _S("best_film",                "film",    "Concept_BestPicture"),
    "outstanding_british_film": _S("outstanding_british_film", "film",    "Concept_BestOutstandingBritishFilm"),
    "best_actor":               _S("best_actor",               "nominee", "Concept_BestMaleLeadActor",        "actor"),
    "best_actress":             _S("best_actress",             "nominee", "Concept_BestFemaleLeadActor",      "actress"),
    "best_supporting_actor":    _S("best_supporting_actor",    "nominee", "Concept_BestMaleSupportingActor",  "supporting_actor"),
    "best_supporting_actress":  _S("best_supporting_actress",  "nominee", "Concept_BestFemaleSupportingActor","supporting_actress"),
    "best_director":            _S("best_director",            "nominee", "Concept_BestDirector",             "director"),
    "best_original_screenplay": _S("best_original_screenplay", "nominee", "Concept_BestOriginalScreenplay",   "original_screenplay"),
    "best_adapted_screenplay":  _S("best_adapted_screenplay",  "nominee", "Concept_BestAdaptedScreenplay",    "adapted_screenplay"),
    "cinematography":           _S("cinematography",           "nominee", "Concept_BestCinematography",       "cinematography"),
    "film_editing":             _S("film_editing",             "nominee", "Concept_BestFilmEditing",          "film_editing"),
}

_SAG_CATEGORY_SPECS: dict[str, CategorySpec] = {
    "best_actor":              _S("best_actor",              "nominee", "Concept_BestMaleLeadActor",        "actor"),
    "best_actress":            _S("best_actress",            "nominee", "Concept_BestFemaleLeadActor",      "actress"),
    "best_supporting_actor":   _S("best_supporting_actor",   "nominee", "Concept_BestMaleSupportingActor",  "supporting_actor"),
    "best_supporting_actress": _S("best_supporting_actress", "nominee", "Concept_BestFemaleSupportingActor","supporting_actress"),
    "best_ensemble_cast":      _S("best_ensemble_cast",      "film",    "Concept_BestEnsembleCast"),
}

_SPECS_BY_SYSTEM: dict[str, dict[str, CategorySpec]] = {
    "oscars": _OSCAR_CATEGORY_SPECS,
    "bafta":  _BAFTA_CATEGORY_SPECS,
    "sag":    _SAG_CATEGORY_SPECS,
}


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _norm_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).upper().strip().split())


def _norm_person_key(value: str | None) -> str:
    normalized = normalize_person_name(value or "")
    return "".join(ch.lower() for ch in normalized if ch.isalnum())


def _local_id(uri: str | None) -> str | None:
    if not uri:
        return None
    text = str(uri)
    return text.split("#", 1)[1] if "#" in text else text


def _person_name(graph: Graph, nominee_uri: str | None) -> str | None:
    if not nominee_uri:
        return None
    full_name = graph.value(URIRef(str(nominee_uri)), MSH.fullName)
    return str(full_name) if full_name is not None else _local_id(nominee_uri)


def _build_concept_index(graph: Graph) -> dict[str, set[str]]:
    """Return {cat_uri → set[concept_uri]} from all msh:realizationOf triples."""
    index: dict[str, set[str]] = {}
    for cat, _, concept in graph.triples((None, MSH.realizationOf, None)):
        cat_str = str(cat)
        if cat_str not in index:
            index[cat_str] = set()
        index[cat_str].add(str(concept))
    return index


# ---------------------------------------------------------------------------
# SPARQL helpers (system short-name lookup)
# ---------------------------------------------------------------------------

_QUERY_NOM_SYSTEM = """\
PREFIX msh: <http://example.org/ontologies/MovieSHACL3#>

SELECT ?nom ?systemShort
WHERE {
  ?nom a msh:Nomination ;
       msh:hasCeremony ?cer .
  ?cer msh:hasAwardSystem ?sys .
  ?sys msh:shortName ?systemShort .
}
"""


def _query_nomination_systems(graph: Graph) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for row in graph.query(_QUERY_NOM_SYSTEM):
        rows.append({"nom": str(row.nom), "system_short": str(row.systemShort)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _list_enrichment_ttls() -> list[Path]:
    enriched_dir = PROJECT_ROOT / "data" / "enriched"
    if not enriched_dir.exists():
        return []
    return sorted(enriched_dir.glob("*.ttl"))


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def _coerce_year_int(df: pd.DataFrame, col: str = "year_film") -> pd.DataFrame:
    if col not in df.columns:
        return df
    out = df.copy()
    years = pd.to_numeric(out[col], errors="coerce")
    bad_mask = years.isna()
    if bad_mask.any():
        bad_count = int(bad_mask.sum())
        warnings.warn(f"Dropping {bad_count} rows with non-integer year_film values", stacklevel=2)
        out = out.loc[~bad_mask].copy()
        years = years.loc[~bad_mask]
    out[col] = years.astype(int)
    return out


def _build_person_history(rows: pd.DataFrame) -> dict[tuple[str, int], tuple[int, int]]:
    history: dict[tuple[str, int], tuple[int, int]] = {}
    if rows.empty:
        return history
    for nominee, grp in rows.groupby("nominee"):
        if nominee is None:
            continue
        grp_sorted = grp.sort_values("year_film")
        years = sorted(grp_sorted["year_film"].astype(int).unique())
        prior_noms = 0
        prior_wins = 0
        for year in years:
            key = (str(nominee), int(year))
            history[key] = (prior_noms, prior_wins)
            year_rows = grp_sorted[grp_sorted["year_film"].astype(int) == int(year)]
            prior_noms += len(year_rows)
            prior_wins += int(year_rows["winner"].astype(int).sum())
    return history


def _compute_film_nominee_history(
    target: pd.DataFrame,
    all_noms: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate prior nomination/win history of individual nominees for film-level categories.

    For each film+year in target, finds all persons listed via msh:hasNominee,
    computes their prior nomination/win history across all categories in the award
    system, then aggregates to film level:
      nominee_previous_noms_sum  — total prior noms summed across nominees
      nominee_previous_noms_max  — max prior noms of any single nominee (star power)
      nominee_previous_noms_avg  — average prior noms across nominees
      nominee_previous_wins_sum  — total prior wins summed across nominees
      nominee_previous_wins_max  — max prior wins of any single nominee
      nominee_previous_wins_avg  — average prior wins across nominees

    Rows without nominees (film-only nominations) get 0 for all columns.
    """
    out = target.copy()
    for col in _NOMINEE_HISTORY_COLS:
        out[col] = 0.0

    person_all = all_noms[all_noms["nominee"].notna()].copy()
    if person_all.empty or target["nominee"].isna().all():
        return out

    hist = _build_person_history(person_all)

    for (film_uri, year), grp in target.groupby(["film_uri", "year_film"]):
        nominees = grp["nominee"].dropna().unique()
        if len(nominees) == 0:
            continue
        noms_list = [hist.get((str(n), int(year)), (0, 0))[0] for n in nominees]
        wins_list = [hist.get((str(n), int(year)), (0, 0))[1] for n in nominees]
        n = len(noms_list)
        idx = (out["film_uri"] == film_uri) & (out["year_film"].astype(int) == int(year))
        out.loc[idx, "nominee_previous_noms_sum"] = sum(noms_list)
        out.loc[idx, "nominee_previous_noms_max"] = max(noms_list)
        out.loc[idx, "nominee_previous_noms_avg"] = sum(noms_list) / n
        out.loc[idx, "nominee_previous_wins_sum"] = sum(wins_list)
        out.loc[idx, "nominee_previous_wins_max"] = max(wins_list)
        out.loc[idx, "nominee_previous_wins_avg"] = sum(wins_list) / n

    return out


_QUERY_GG_BEST_FILM = """\
PREFIX msh: <http://example.org/ontologies/MovieSHACL3#>
SELECT ?film ?ceremonyYear ?cat WHERE {
  ?nom a msh:Nomination ;
       msh:hasFilm ?film ;
       msh:hasCategory ?cat ;
       msh:hasCeremony ?cer ;
       msh:winner true .
  ?cer msh:yearCeremony ?ceremonyYear .
  ?cer msh:hasAwardSystem ?sys .
  ?sys msh:shortName "GOLDEN_GLOBES" .
  FILTER(?cat IN (
    msh:Category_golden_globes_Best_Motion_Picture_Drama,
    msh:Category_golden_globes_Best_Motion_Picture_Musical_or_Comedy
  ))
}
"""

_QUERY_BAFTA_BEST_FILM = """\
PREFIX msh: <http://example.org/ontologies/MovieSHACL3#>
SELECT ?film ?ceremonyYear WHERE {
  ?nom a msh:Nomination ;
       msh:hasFilm ?film ;
       msh:hasCategory msh:Category_bafta_Best_Film ;
       msh:hasCeremony ?cer ;
       msh:winner true .
  ?cer msh:yearCeremony ?ceremonyYear .
}
"""

_QUERY_SAG_ENSEMBLE_WINNERS = """\
PREFIX msh: <http://example.org/ontologies/MovieSHACL3#>
SELECT ?film ?ceremonyYear WHERE {
  ?nom a msh:Nomination ;
       msh:hasFilm ?film ;
       msh:hasCategory msh:Category_sag_Outstanding_Performance_by_a_Cast_in_a_Motion_Picture ;
       msh:hasCeremony ?cer ;
       msh:winner true .
  ?cer msh:yearCeremony ?ceremonyYear .
}
"""

_GG_DRAMA_CAT = f"{MSH_BASE}Category_golden_globes_Best_Motion_Picture_Drama"
_GG_COMEDY_CAT = f"{MSH_BASE}Category_golden_globes_Best_Motion_Picture_Musical_or_Comedy"


def _compute_best_film_precursors(graph: Graph, df: pd.DataFrame) -> pd.DataFrame:
    """Add globe_drama_winner, globe_comedy_winner, bafta_best_film_winner columns."""
    out = df.copy()
    for col in _BEST_FILM_PRECURSOR_COLS:
        out[col] = 0

    # Golden Globe category-specific winners.
    gg_rows = [
        {"film_uri": str(r.film), "ceremony_year": int(str(r.ceremonyYear)[:4]), "cat": str(r.cat)}
        for r in graph.query(_QUERY_GG_BEST_FILM)
    ]
    gg_df = pd.DataFrame(gg_rows) if gg_rows else pd.DataFrame(columns=["film_uri", "ceremony_year", "cat"])

    # BAFTA Best Film winners.
    bafta_rows = [
        {"film_uri": str(r.film), "ceremony_year": int(str(r.ceremonyYear)[:4])}
        for r in graph.query(_QUERY_BAFTA_BEST_FILM)
    ]
    bafta_df = pd.DataFrame(bafta_rows) if bafta_rows else pd.DataFrame(columns=["film_uri", "ceremony_year"])

    for idx, row in out.iterrows():
        film_uri = row.get("film_uri")
        if not film_uri:
            continue
        year = int(row["year_film"])

        # GG drama/comedy.
        if not gg_df.empty:
            gg_match = gg_df[
                (gg_df["film_uri"] == film_uri)
                & (gg_df["ceremony_year"] >= year - 1)
                & (gg_df["ceremony_year"] <= year + 2)
            ]
            for _, gg_row in gg_match.iterrows():
                if gg_row["cat"] == _GG_DRAMA_CAT:
                    out.at[idx, "globe_drama_winner"] = 1
                elif gg_row["cat"] == _GG_COMEDY_CAT:
                    out.at[idx, "globe_comedy_winner"] = 1

        # BAFTA Best Film.
        if not bafta_df.empty:
            bafta_match = bafta_df[
                (bafta_df["film_uri"] == film_uri)
                & (bafta_df["ceremony_year"] >= year - 1)
                & (bafta_df["ceremony_year"] <= year + 2)
            ]
            if not bafta_match.empty:
                out.at[idx, "bafta_best_film_winner"] = 1

    return out


def _compute_sag_ensemble_feature(graph: Graph, df: pd.DataFrame) -> pd.DataFrame:
    """Add sag_ensemble_winner column: 1 if the film won SAG Outstanding Cast."""
    out = df.copy()
    out["sag_ensemble_winner"] = 0

    rows = [
        {"film_uri": str(r.film), "ceremony_year": int(str(r.ceremonyYear)[:4])}
        for r in graph.query(_QUERY_SAG_ENSEMBLE_WINNERS)
    ]
    if not rows:
        return out

    sag_df = pd.DataFrame(rows)
    for idx, row in out.iterrows():
        film_uri = row.get("film_uri")
        if not film_uri:
            continue
        year = int(row["year_film"])
        match = sag_df[
            (sag_df["film_uri"] == film_uri)
            & (sag_df["ceremony_year"] >= year - 1)
            & (sag_df["ceremony_year"] <= year + 2)
        ]
        if not match.empty:
            out.at[idx, "sag_ensemble_winner"] = 1
    return out


def _compute_cast_member_history(
    target_df: pd.DataFrame, all_noms: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate prior SAG nomination/win history of individually-nominated cast members.

    For each film+year in target_df, finds all persons nominated in SAG acting
    categories for that film that year, then sums their personal prior histories.
    Returns target_df with three new columns:
      cast_previous_noms_sum   - sum of prior SAG noms across cast
      cast_previous_wins_sum   - sum of prior SAG wins across cast
      cast_star_power          - max prior noms of any single cast member
    """
    out = target_df.copy()
    out["cast_previous_noms_sum"] = 0
    out["cast_previous_wins_sum"] = 0
    out["cast_star_power"] = 0

    # Filter to SAG acting nominations only.
    acting_noms = all_noms[
        all_noms["concept_uris"].apply(
            lambda s: bool(s & _SAG_ACTING_CONCEPTS)
        )
    ].copy()
    if acting_noms.empty:
        return out

    # Build person history across all SAG acting categories.
    person_hist = _build_person_history(acting_noms)

    for idx, row in out.iterrows():
        film_uri = row.get("film_uri")
        year = int(row["year_film"])
        # Find acting nominees for this film this year.
        cast_noms = acting_noms[
            (acting_noms["film_uri"] == film_uri)
            & (acting_noms["year_film"].astype(int) == year)
            & (acting_noms["nominee"].notna())
        ]
        if cast_noms.empty:
            continue
        noms_list = []
        wins_list = []
        for nominee_uri in cast_noms["nominee"].unique():
            prior_noms, prior_wins = person_hist.get((str(nominee_uri), year), (0, 0))
            noms_list.append(prior_noms)
            wins_list.append(prior_wins)
        out.at[idx, "cast_previous_noms_sum"] = sum(noms_list)
        out.at[idx, "cast_previous_wins_sum"] = sum(wins_list)
        out.at[idx, "cast_star_power"] = max(noms_list) if noms_list else 0

    return out


def _add_nominee_label_columns(df: pd.DataFrame, graph: Graph) -> pd.DataFrame:
    out = df.copy()
    out["candidate_id"] = out["nominee"].apply(_local_id)
    out["candidate_name"] = out["nominee"].apply(lambda uri: _person_name(graph, uri))
    return out


def _ensure_required_columns(df: pd.DataFrame, row_type: str) -> pd.DataFrame:
    out = df.copy()
    for col in _REQUIRED_COMMON_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    if row_type == "nominee":
        for col in _REQUIRED_NOMINEE_COLS:
            if col not in out.columns:
                out[col] = pd.NA
    return out


def _aggregate_film_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ordered = df.sort_values(["film_uri", "year_film", "winner", "nom"], ascending=[True, True, False, True])
    agg_map: dict[str, str] = {
        c: "first" for c in ordered.columns if c not in {"film_uri", "year_film", "winner"}
    }
    agg_map["winner"] = "max"
    out = ordered.groupby(["film_uri", "year_film"], as_index=False).agg(agg_map)
    return out


def _aggregate_nominee_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ordered = df.sort_values(["nominee", "year_film", "winner", "nom"], ascending=[True, True, False, True])
    agg_map: dict[str, str] = {
        c: "first" for c in ordered.columns if c not in {"nominee", "year_film", "winner"}
    }
    agg_map["winner"] = "max"
    out = ordered.groupby(["nominee", "year_film"], as_index=False).agg(agg_map)
    return out


# ---------------------------------------------------------------------------
# Concept-driven cross-system precursor flags
# ---------------------------------------------------------------------------

def _attach_concept_film_precursor_flags(
    df: pd.DataFrame,
    all_noms: pd.DataFrame,
    concept_uri: str,
    target_system_key: str,
    col_prefix: str,
) -> pd.DataFrame:
    """Attach film-level winner/nominee flags from precursor systems sharing the same concept.

    Used for film-level categories (e.g. screenplay) where matching is by film URI
    rather than by person name.  Adds:
      {system_key}_{col_prefix}_nominee
      {system_key}_{col_prefix}_winner
    """
    out = df.copy()
    target_short = _SYSTEM_KEYS[target_system_key]

    precursor_noms = all_noms[
        all_noms["concept_uris"].apply(lambda s: concept_uri in s)
        & (all_noms["system_short"] != target_short)
        & all_noms["film_uri"].notna()
    ].copy()

    if precursor_noms.empty:
        return out

    short_to_key = {v: k for k, v in _SYSTEM_KEYS.items()}

    for system_short, sys_df in precursor_noms.groupby("system_short"):
        system_key = short_to_key.get(str(system_short), str(system_short).lower())

        nominee_set: set[tuple[int, str]] = {
            (int(y), str(f))
            for y, f in zip(sys_df["year_film"], sys_df["film_uri"])
            if f
        }
        winner_set: set[tuple[int, str]] = {
            (int(y), str(f))
            for y, f, w in zip(sys_df["year_film"], sys_df["film_uri"], sys_df["winner"])
            if f and w
        }

        nom_col = f"{system_key}_{col_prefix}_nominee"
        win_col = f"{system_key}_{col_prefix}_winner"
        out[nom_col] = [
            1 if (int(y), str(f)) in nominee_set else 0
            for y, f in zip(out["year_film"].astype(int), out["film_uri"])
        ]
        out[win_col] = [
            1 if (int(y), str(f)) in winner_set else 0
            for y, f in zip(out["year_film"].astype(int), out["film_uri"])
        ]

    return out


def _attach_concept_precursor_flags(
    df: pd.DataFrame,
    all_noms: pd.DataFrame,
    graph: Graph,
    concept_uri: str,
    target_system_key: str,
    col_prefix: str,
) -> pd.DataFrame:
    """Attach nominee/winner flags from every precursor system sharing the same concept.

    For each system other than target_system_key that has nominations whose
    category realizationOf == concept_uri, adds two columns:
      {system_key}_{col_prefix}_nominee
      {system_key}_{col_prefix}_winner

    The lookup is purely graph-driven via the msh:realizationOf links in
    category_alignments.ttl — no string matchers required.

    For person-level precursor nominations (hasNominee present) matching is
    done by normalized person name.  For film-level nominations (no hasNominee,
    e.g. BAFTA Original Music) matching falls back to film URI, which allows
    craft categories whose precursors track the film rather than the individual
    technician to still produce useful precursor flags.
    """
    out = df.copy()
    target_short = _SYSTEM_KEYS[target_system_key]

    # Include ALL precursor nominations for this concept — film-level and
    # person-level alike (the nominee.notna() filter is removed vs old version).
    precursor_noms = all_noms[
        all_noms["concept_uris"].apply(lambda s: concept_uri in s)
        & (all_noms["system_short"] != target_short)
    ].copy()

    if precursor_noms.empty:
        return out

    precursor_noms["prec_cand_name"] = precursor_noms["nominee"].apply(
        lambda uri: _person_name(graph, uri) if uri else None
    )
    precursor_noms["prec_cand_key"] = precursor_noms["prec_cand_name"].apply(_norm_person_key)
    precursor_noms["prec_year"] = precursor_noms["year_film"].astype(int)

    out["prec_cand_key"] = out["candidate_name"].apply(_norm_person_key)

    short_to_key = {v: k for k, v in _SYSTEM_KEYS.items()}

    for system_short, sys_df in precursor_noms.groupby("system_short"):
        system_key = short_to_key.get(str(system_short), str(system_short).lower())
        nom_col = f"{system_key}_{col_prefix}_nominee"
        win_col = f"{system_key}_{col_prefix}_winner"

        # Person-level: match by normalized candidate name.
        person_df = sys_df[sys_df["nominee"].notna()]
        nom_by_name: set[tuple[int, str]] = {
            (int(y), str(k))
            for y, k in zip(person_df["prec_year"], person_df["prec_cand_key"])
            if k
        }
        win_by_name: set[tuple[int, str]] = {
            (int(y), str(k))
            for y, k, w in zip(person_df["prec_year"], person_df["prec_cand_key"], person_df["winner"])
            if k and w
        }

        # Film-level: match by film URI (covers craft categories where the
        # precursor system records the film, not the individual technician).
        film_df = sys_df[sys_df["nominee"].isna()]
        nom_by_film: set[tuple[int, str]] = {
            (int(y), str(f))
            for y, f in zip(film_df["prec_year"], film_df["film_uri"])
            if f
        }
        win_by_film: set[tuple[int, str]] = {
            (int(y), str(f))
            for y, f, w in zip(film_df["prec_year"], film_df["film_uri"], film_df["winner"])
            if f and w
        }

        out[nom_col] = [
            1 if (
                (int(y), k) in nom_by_name
                or (int(y), str(f)) in nom_by_film
            ) else 0
            for y, k, f in zip(out["year_film"].astype(int), out["prec_cand_key"], out["film_uri"])
        ]
        out[win_col] = [
            1 if (
                (int(y), k) in win_by_name
                or (int(y), str(f)) in win_by_film
            ) else 0
            for y, k, f in zip(out["year_film"].astype(int), out["prec_cand_key"], out["film_uri"])
        ]

    out = out.drop(columns=["prec_cand_key"])
    return out


# ---------------------------------------------------------------------------
# CategoryExtractor
# ---------------------------------------------------------------------------

class CategoryExtractor:
    """Concept-driven extractor for any registered award system category."""

    def __init__(self, category: str, award_system: str = "oscars"):
        system = str(award_system).strip().lower()
        if system not in _SPECS_BY_SYSTEM:
            raise ValueError(
                f"Unsupported award_system: {award_system!r}. "
                f"Supported: {sorted(_SPECS_BY_SYSTEM)}"
            )
        self._award_system = system
        specs_map = _SPECS_BY_SYSTEM[system]
        alias = str(category).strip().lower()
        if alias not in specs_map:
            allowed = ", ".join(sorted(specs_map))
            raise ValueError(
                f"Unsupported category alias for '{system}': {category!r}. "
                f"Supported: {allowed}"
            )
        self._spec = specs_map[alias]
        self._result: pd.DataFrame | None = None

    @property
    def label_cols(self) -> list[str]:
        return ["film"] if self._spec.row_type == "film" else ["candidate_name", "film"]

    @property
    def row_type(self) -> str:
        return self._spec.row_type

    def extract(self) -> pd.DataFrame:
        graph = load_foundation_graph(include_ontology=True)

        for ttl_path in _list_enrichment_ttls():
            graph.parse(str(ttl_path), format="turtle")

        # Canonical features (Oscar Best Picture pipeline — used for Oscar BP only,
        # but we still call it to populate extract_category_features data).
        extract_canonical_features(graph)

        # Build enrichment lookup over ALL films in the graph so non-Oscar
        # films (BAFTA-only, SAG-only) still get enrichment features.
        from rdflib import RDF, URIRef as _URIRef
        _MSH_Film = _URIRef(f"{MSH_BASE}Film")
        all_film_uris = [str(s) for s in graph.subjects(RDF.type, _MSH_Film)]
        all_films_stub = pd.DataFrame({"film_uri": all_film_uris, "winner": False})
        enriched_df = extract_graph_enrichment(graph, all_films_stub)

        # Query all nominations (now includes cat_uri).
        all_noms = _query_nominations(graph)
        if all_noms.empty:
            out = _ensure_required_columns(pd.DataFrame(), self.row_type)
            self._result = out
            return out

        all_noms = _coerce_year_int(all_noms)

        # Attach system short name.
        system_df = _query_nomination_systems(graph)
        if not system_df.empty:
            all_noms = all_noms.merge(system_df, on="nom", how="left")
        else:
            all_noms["system_short"] = None

        # Attach concept URIs from the category_alignments graph.
        concept_index = _build_concept_index(graph)
        all_noms["concept_uris"] = all_noms["cat_uri"].apply(
            lambda uri: concept_index.get(str(uri), frozenset())
        )

        # Filter to target system + target concept.
        target_short = _SYSTEM_KEYS[self._award_system]
        concept_uri = self._spec.concept_uri

        system_noms = all_noms[all_noms["system_short"] == target_short].copy()
        if system_noms.empty:
            out = _ensure_required_columns(pd.DataFrame(), self.row_type)
            self._result = out
            return out

        target = system_noms[
            system_noms["concept_uris"].apply(lambda s: concept_uri in s)
        ].copy()
        if target.empty:
            out = _ensure_required_columns(pd.DataFrame(), self.row_type)
            self._result = out
            return out

        # Per-film category nomination counts (acting, directing, etc.).
        film_cat = extract_category_features(system_noms)
        target = target.merge(film_cat, on=["film", "year_film"], how="left")
        for col in _FILM_CAT_COLS:
            if col not in target.columns:
                target[col] = 0
        target[_FILM_CAT_COLS] = target[_FILM_CAT_COLS].fillna(0).astype(int)

        # Film-level precursor features (cross-system film winners).
        target = extract_precursor_features(graph, target)
        excl_col = _FILM_PRECURSOR_EXCL.get(self._award_system)
        if excl_col and excl_col in target.columns:
            target = target.drop(columns=[excl_col])
        active_precursor_cols = [c for c in _ALL_FILM_PRECURSOR_COLS if c != excl_col]
        for col in active_precursor_cols:
            if col not in target.columns:
                target[col] = 0
        target[active_precursor_cols] = target[active_precursor_cols].fillna(0).astype(int)

        # Merge film enrichment features.
        base_for_enrichment = target[["film_uri", "film", "year_film", "winner"]].drop_duplicates().copy()
        shared_base_cols = set(base_for_enrichment.columns)
        enrichment_cols = [
            c for c in enriched_df.columns
            if c not in shared_base_cols and c not in target.columns
        ]
        if enrichment_cols:
            enrichment_lookup = enriched_df.reindex(
                columns=["film_uri", *enrichment_cols]
            ).drop_duplicates("film_uri")
            target = target.merge(enrichment_lookup, on="film_uri", how="left")

        # Nominee-level processing.
        if self.row_type == "nominee":
            out = target[target["nominee"].notna()].copy()
            out = _add_nominee_label_columns(out, graph)

            # Person history within this category and across all categories.
            person_all = system_noms[system_noms["nominee"].notna()].copy()
            person_cat = out.copy()
            all_hist = _build_person_history(person_all)
            cat_hist = _build_person_history(person_cat)
            prefix = self._spec.col_prefix or self._spec.alias

            out["previous_all_nominations"] = out.apply(
                lambda r: all_hist.get((str(r["nominee"]), int(r["year_film"])), (0, 0))[0],
                axis=1,
            )
            out["previous_all_wins"] = out.apply(
                lambda r: all_hist.get((str(r["nominee"]), int(r["year_film"])), (0, 0))[1],
                axis=1,
            )
            out[f"previous_{prefix}_nominations"] = out.apply(
                lambda r: cat_hist.get((str(r["nominee"]), int(r["year_film"])), (0, 0))[0],
                axis=1,
            )
            out[f"previous_{prefix}_wins"] = out.apply(
                lambda r: cat_hist.get((str(r["nominee"]), int(r["year_film"])), (0, 0))[1],
                axis=1,
            )

            # Cross-system concept precursor flags.
            out = _attach_concept_precursor_flags(
                out, all_noms, graph,
                concept_uri=concept_uri,
                target_system_key=self._award_system,
                col_prefix=prefix,
            )

            out = _aggregate_nominee_rows(out)

        else:
            # Ensemble-cast-specific enrichment.
            if self._spec.alias == "best_ensemble_cast":
                target = _compute_best_film_precursors(graph, target)
                target = _compute_cast_member_history(target, all_noms)
            # SAG ensemble cast is the film-level Best Picture proxy at SAG.
            # Compute it as a dedicated feature for Oscar best_picture so RFECV
            # sees a clean signal instead of the noisy all-SAG-winners sag_winner.
            if self._award_system == "oscars" and self._spec.alias == "best_picture":
                target = _compute_sag_ensemble_feature(graph, target)
            # Aggregate nominee histories (sum, max/star-power, avg) for all
            # film-level categories that have individual nominees (COLLABORATION
            # or PERSON). Film-only categories (documentary, animated, etc.)
            # get 0s since they have no hasNominee triples.
            target = _compute_film_nominee_history(target, all_noms)
            # For film-level categories with a col_prefix (e.g. screenplay),
            # attach category-specific precursor flags from other systems via
            # film URI matching (analogous to _attach_concept_precursor_flags
            # but without person-name lookup).
            if self._spec.col_prefix is not None:
                target = _attach_concept_film_precursor_flags(
                    target, all_noms,
                    concept_uri=concept_uri,
                    target_system_key=self._award_system,
                    col_prefix=self._spec.col_prefix,
                )
            out = _aggregate_film_rows(target)

        out = _coerce_year_int(out)
        out = _ensure_required_columns(out, self.row_type)
        out["winner"] = out["winner"].fillna(False).astype(bool)

        lead_cols = list(_REQUIRED_COMMON_COLS)
        if self.row_type == "nominee":
            lead_cols.extend(_REQUIRED_NOMINEE_COLS)
        tail_cols = [c for c in out.columns if c not in lead_cols]
        ordered_cols = [c for c in lead_cols if c in out.columns] + tail_cols
        out = out[ordered_cols].sort_values(["year_film", "winner"], ascending=[True, False]).reset_index(drop=True)

        self._result = out
        return out


__all__ = ["CategoryExtractor"]
