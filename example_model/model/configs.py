"""Feature set configurations and model factory for Best Picture prediction."""

from __future__ import annotations

# 10 named feature sets (A–J) from the modelling notebook.
# Values are list[str] of column names, except Config F which is the sentinel "RFECV".
FEATURE_CONFIGS: dict[str, list[str] | str] = {
    "A: Baseline (4-feat)": [
        "other_noms", "imdb_rating", "previous_nominations_max", "genre_horror",
    ],
    "B: Baseline + log_revenue": [
        "other_noms", "imdb_rating", "previous_nominations_max", "genre_horror", "log_revenue",
    ],
    "C: Original 6-feat": [
        "previous_nominations_max", "other_noms", "vote_average", "log_budget",
        "log_revenue", "genre_horror",
    ],
    "D: Category decomposition": [
        "acting_noms", "directing_nom", "editing_nom", "writing_noms",
        "technical_noms", "imdb_rating", "genre_horror",
    ],
    "E: Above-line focus": [
        "above_line_noms", "editing_nom", "imdb_rating", "log_revenue", "genre_horror",
    ],
    "F: RFECV auto-select": "RFECV",
    "G: Curated broad": [
        "acting_noms", "directing_nom", "editing_nom", "writing_noms", "other_noms",
        "imdb_rating", "vote_average", "log_revenue", "has_prestige_company",
    ],
    "H: Minimal prestige": [
        "above_line_noms", "editing_nom", "imdb_rating",
    ],
    "I: Precursor awards": [
        "pga_winner", "sag_winner", "globe_winner", "dga_winner", "bafta_winner",
        "imdb_rating", "above_line_noms", "editing_nom",
    ],
    "J: Precursor + baseline": [
        "num_precursor_wins", "other_noms", "imdb_rating",
        "previous_nominations_max", "genre_horror",
    ],
}

MODEL_TYPES = ["LR", "RF", "GB", "LR+RF"]

# Columns that are identifiers or targets — never used as features.
EXCLUDE_COLUMNS = frozenset({
    "film", "year_film", "tmdb_id", "imdb_id",
    "nom", "nominee", "nominee_type", "winner",
})


def create_estimator(name: str):
    """Factory returning a fresh sklearn estimator by short name.

    Raises ValueError for 'LR+RF' (ensemble handled in trainer) and unknown names.
    """
    if name == "LR":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced",
        )
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced", random_state=42,
        )
    if name == "GB":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42,
        )
    if name == "LR+RF":
        raise ValueError("LR+RF ensemble is handled in the trainer, not as a single estimator")
    raise ValueError(f"Unknown model type: {name!r}")
