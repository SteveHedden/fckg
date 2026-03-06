"""
Conceptual feature groupings for award prediction models.

Purpose
───────
Pure RFECV picks whatever is statistically predictive but can select
multiple features that measure the same underlying concept (e.g. both
budget and log_budget for financial scale, or both rt_score and
metacritic_score for critical reception). This module enforces a
"one representative per concept" rule before RFECV runs.

How it works
────────────
1. For each conceptual group, find which candidate features are present
   in the training data.
2. Among the present candidates, pick the one with the highest
   univariate AUC vs. the winner column.  If coverage is too low
   (< 20 % non-null), the feature is skipped.
3. Precursor nominee/winner pairs for the same ceremony are auto-grouped:
   <system>_<category>_winner is always preferred over _nominee.
4. The curated pool (one per group + all ungrouped columns) is returned
   for RFECV to run on.

Adding a new group
──────────────────
Add an entry to FEATURE_GROUPS.  The list order is the tiebreak
preference if two features have identical AUC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ── Conceptual groups ─────────────────────────────────────────────────────────
# Within each group, only the single best-performing feature is kept.
# Candidates listed in preference order (used as tiebreak).

FEATURE_GROUPS: dict[str, list[str]] = {
    # Box office / financial scale
    "financial": [
        "revenue", "log_revenue",
        "budget", "log_budget",
        "budget_revenue_ratio",
    ],
    # Audience quality signal (ratings)
    "audience_rating": [
        "vote_average",
        "imdb_rating",
    ],
    # Audience scale signal (how many people engaged)
    "audience_scale": [
        "vote_count",
        "popularity",
    ],
    # Professional critic aggregators (redundant with each other)
    "critical_reception": [
        "rt_score",
        "metacritic_score",
    ],
    # Film's prior nomination count across ceremonies (sum/max/avg are redundant)
    "film_history_noms": [
        "previous_nominations_sum",
        "previous_nominations_max",
        "previous_nominations_avg",
    ],
    # Film's prior win count
    "film_history_wins": [
        "previous_wins_sum",
        "previous_wins_max",
    ],
    # Broad awards circuit recognition
    # (composite preferred over individual festival flags)
    "festival_circuit": [
        "big_three_festival",
        "num_wikidata_awards",
        "palme_dor",
        "golden_lion",
        "golden_bear",
    ],
}

# ── Precursor ceremony logic ──────────────────────────────────────────────────
# Film-level precursor flags (pga_winner, dga_winner, etc.) are kept as
# singletons — each ceremony is a distinct signal.
#
# Nominee-level precursor pairs (<system>_<category>_nominee /
# <system>_<category>_winner) are auto-detected by _find_precursor_pairs and
# reduced to the winner column (winner implies nominee; both measure the same
# event, nominee is redundant).
#
# Category-aware precursor importance (informational — RFECV decides weight):
#   best_director    → dga_director_winner is the dominant signal
#   best_actor/actress → oscars_actor_winner, bafta_actor_winner, globe_actor_winner
#   best_supporting  → globe_supporting_winner, oscars_supporting_winner
#   best_ensemble    → globe_drama_winner, globe_comedy_winner, bafta_best_film_winner
#   best_picture     → pga_winner, dga_winner are historically strongest
#
# These are not hard-coded exclusions — all precursor winner columns enter the
# RFECV pool and the model learns which are predictive for each category.

# Columns that are always kept as singletons (one per concept by definition).
# These are never merged into a group even if patterns match.
_ALWAYS_SINGLETON_PREFIXES = (
    "genre_",
    "country_",
    "rated_",
    "pga_winner",
    "dga_winner",
    "sag_winner",
    "bafta_winner",
    "globe_winner",
    "globe_drama_winner",
    "globe_comedy_winner",
    "bafta_best_film_winner",
    "acting_noms",
    "directing_nom",
    "writing_noms",
    "editing_nom",
    "cinematography_nom",
    "above_line_noms",
    "technical_noms",
    "other_noms",
    "music_noms",
    "english_language",
    "release_month",
    "runtime",
    "has_collection",
    "cast_previous",
    "previous_all_nominations",
    "previous_all_wins",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _univariate_auc(series: pd.Series, y: pd.Series) -> float:
    """Symmetric AUC (always ≥ 0.5) for a single feature vs binary target."""
    mask = series.notna()
    if mask.sum() < max(10, 0.2 * len(series)):
        return 0.5          # too sparse — skip
    if y[mask].nunique() < 2:
        return 0.5
    try:
        auc = roc_auc_score(y[mask].astype(int), series[mask])
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def _find_precursor_pairs(cols: list[str]) -> dict[str, list[str]]:
    """Detect *_nominee / *_winner column pairs and return as groups.

    For each pair the winner column is listed first (preferred).
    """
    col_set = set(cols)
    nominee_cols = [c for c in cols if c.endswith("_nominee")]
    pairs: dict[str, list[str]] = {}
    for nom_col in nominee_cols:
        win_col = nom_col[:-len("_nominee")] + "_winner"
        if win_col in col_set:
            group_key = f"precursor_pair__{nom_col[:-len('_nominee')]}"
            pairs[group_key] = [win_col, nom_col]   # winner preferred
    return pairs


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GroupSelectionReport:
    """Summary of the group-based reduction step."""
    selected: list[str]                         # curated feature pool for RFECV
    group_picks: dict[str, str] = field(default_factory=dict)   # group → chosen col
    group_aucs: dict[str, float] = field(default_factory=dict)  # group → best AUC
    dropped: list[str] = field(default_factory=list)            # cols eliminated

    def print_summary(self) -> None:
        print(f"\nConceptual reduction: {len(self.selected)} features "
              f"({len(self.dropped)} dropped as redundant within groups)\n")
        if self.group_picks:
            print("  Group representatives selected:")
            for grp, col in sorted(self.group_picks.items()):
                auc = self.group_aucs.get(grp, 0.5)
                print(f"    {grp:<35s}  {col:<40s}  AUC={auc:.3f}")
        if self.dropped:
            print(f"\n  Dropped (redundant within group): {self.dropped}")


# ── Category-aware ceremony groups ────────────────────────────────────────────
# For each target category, a ceremony (BAFTA, Globes) may produce both a
# film-level winner signal (bafta_winner, globe_winner) and a category-specific
# winner signal (bafta_actress_winner, golden_globes_actress_winner).  These
# measure the same underlying event and only one should enter RFECV.
#
# Within each group, candidates are listed in preference order (tiebreak).
# The category-specific winner is always listed first so it wins on ties.
# AUC decides when there is a clear empirical difference.
#
# These groups supersede auto-detected precursor pairs for the same columns.

CATEGORY_CEREMONY_GROUPS: dict[str, dict[str, list[str]]] = {
    "best_actor": {
        "bafta_signal": [
            "bafta_actor_winner", "bafta_actor_nominee", "bafta_winner",
        ],
        "globes_signal": [
            "golden_globes_actor_winner", "golden_globes_actor_nominee",
            "globe_winner", "globe_drama_winner", "globe_comedy_winner",
        ],
    },
    "best_actress": {
        "bafta_signal": [
            "bafta_actress_winner", "bafta_actress_nominee", "bafta_winner",
        ],
        "globes_signal": [
            "golden_globes_actress_winner", "golden_globes_actress_nominee",
            "globe_winner", "globe_drama_winner", "globe_comedy_winner",
        ],
    },
    "best_supporting_actor": {
        "bafta_signal": [
            "bafta_supporting_actor_winner", "bafta_supporting_actor_nominee",
            "bafta_winner",
        ],
        "globes_signal": [
            "golden_globes_supporting_actor_winner",
            "golden_globes_supporting_actor_nominee",
            "globe_winner",
        ],
    },
    "best_supporting_actress": {
        "bafta_signal": [
            "bafta_supporting_actress_winner", "bafta_supporting_actress_nominee",
            "bafta_winner",
        ],
        "globes_signal": [
            "golden_globes_supporting_actress_winner",
            "golden_globes_supporting_actress_nominee",
            "globe_winner",
        ],
    },
    "best_director": {
        "bafta_signal": [
            "bafta_director_winner", "bafta_director_nominee", "bafta_winner",
        ],
        "globes_signal": [
            "golden_globes_director_winner", "golden_globes_director_nominee",
            "globe_winner", "globe_drama_winner", "globe_comedy_winner",
        ],
    },
    "best_picture": {
        "bafta_signal": ["bafta_best_film_winner", "bafta_winner"],
        "globes_signal": [
            "globe_drama_winner", "globe_comedy_winner", "globe_winner",
        ],
    },
    # Ensemble cast is a film-level category — same ceremony grouping as best_picture
    "best_ensemble_cast": {
        "bafta_signal": ["bafta_best_film_winner", "bafta_winner"],
        "globes_signal": [
            "globe_drama_winner", "globe_comedy_winner", "globe_winner",
        ],
    },
}


# ── Main entry point ──────────────────────────────────────────────────────────

def reduce_to_group_representatives(
    train_df: pd.DataFrame,
    candidate_cols: list[str],
    winner_col: str = "winner",
    category: str | None = None,
) -> GroupSelectionReport:
    """Return a curated feature pool with at most one feature per concept group.

    Parameters
    ----------
    train_df       : training DataFrame containing candidate_cols and winner_col
    candidate_cols : full list of numeric feature columns to consider
    winner_col     : binary target column name
    category       : target category alias (e.g. "best_actress").  When
                     provided, CATEGORY_CEREMONY_GROUPS entries are used to
                     group ceremony-level signals (e.g. bafta_actress_winner
                     vs bafta_winner) so the most predictive one is kept.
                     Columns covered by these groups are excluded from
                     auto-detected precursor pairs to avoid double-grouping.

    Returns
    -------
    GroupSelectionReport with .selected = curated pool ready for RFECV
    """
    y = train_df[winner_col].astype(int)
    col_set = set(candidate_cols)

    # Category-aware groups take precedence for ceremony-level signals.
    category_groups: dict[str, list[str]] = (
        CATEGORY_CEREMONY_GROUPS.get(category, {}) if category else {}
    )
    # Columns covered by category-aware groups are excluded from auto-detection
    # so they don't appear in two competing groups simultaneously.
    category_covered: set[str] = {
        c for grp_cols in category_groups.values() for c in grp_cols
    }

    # Build the full group map:
    #   1. Static conceptual groups (financial, ratings, etc.)
    #   2. Category-aware ceremony groups (bafta_signal, globes_signal)
    #   3. Auto-detected precursor pairs for columns NOT covered above
    all_groups: dict[str, list[str]] = {}
    all_groups.update(FEATURE_GROUPS)
    all_groups.update(category_groups)
    remaining_for_autopairs = [c for c in candidate_cols if c not in category_covered]
    all_groups.update(_find_precursor_pairs(remaining_for_autopairs))

    # Track which cols are claimed by a group
    grouped_cols: set[str] = set()
    group_picks: dict[str, str] = {}
    group_aucs: dict[str, float] = {}
    dropped: list[str] = []

    for group_name, candidates in all_groups.items():
        present = [c for c in candidates if c in col_set]
        if len(present) == 0:
            continue
        if len(present) == 1:
            # Only one candidate present — keep it, no elimination needed
            grouped_cols.add(present[0])
            continue

        # Multiple present — pick best by AUC
        aucs = {c: _univariate_auc(train_df[c], y) for c in present}
        # Sort by AUC desc, then by original candidate order as tiebreak
        best = sorted(present, key=lambda c: (-aucs[c], candidates.index(c)))[0]

        group_picks[group_name] = best
        group_aucs[group_name] = aucs[best]
        grouped_cols.update(present)
        dropped.extend([c for c in present if c != best])

    # Everything not claimed by any group is a singleton — keep as-is
    singletons = [c for c in candidate_cols if c not in grouped_cols]

    # Final pool: group winners + singletons, preserving original column order
    winners_set = set(group_picks.values()) | (grouped_cols - set(dropped))
    selected = [c for c in candidate_cols if c in winners_set or c in singletons]

    return GroupSelectionReport(
        selected=selected,
        group_picks=group_picks,
        group_aucs=group_aucs,
        dropped=dropped,
    )
