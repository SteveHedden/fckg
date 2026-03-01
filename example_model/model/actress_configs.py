"""Feature set configurations for Best Actress prediction."""

from __future__ import annotations

from .configs import MODEL_TYPES, create_estimator

FEATURE_CONFIGS_ACTRESS: dict[str, list[str] | str] = {
    "A: Canonical prior history": [
        "previous_actress_nominations",
        "previous_actress_wins",
        "previous_all_nominations",
        "previous_all_wins",
        "directing_nom",
        "writing_noms",
        "other_noms",
    ],
    "B: Canonical + ratings": [
        "previous_actress_nominations",
        "previous_all_nominations",
        "directing_nom",
        "writing_noms",
        "imdb_rating",
        "vote_average",
    ],
    "C: Canonical + market + SAG": [
        "previous_actress_nominations",
        "previous_all_wins",
        "above_line_noms",
        "editing_nom",
        "imdb_rating",
        "log_revenue",
        "sag_actress_nominee",
        "sag_actress_winner",
    ],
    "D: RFECV auto-select": "RFECV",
}

EXCLUDE_COLUMNS_ACTRESS = frozenset({
    "candidate_id", "candidate_name", "film", "year_film",
    "tmdb_id", "imdb_id", "nom", "winner",
})

__all__ = [
    "FEATURE_CONFIGS_ACTRESS",
    "EXCLUDE_COLUMNS_ACTRESS",
    "MODEL_TYPES",
    "create_estimator",
]
