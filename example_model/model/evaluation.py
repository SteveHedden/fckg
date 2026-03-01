"""Result dataclasses and metric computation for Best Picture prediction."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class YearResult:
    """Result of a single backtest year."""

    year: int
    top_pick: str
    actual_winner: str
    winner_rank: int
    winner_prob: float
    correct: bool


@dataclass
class BacktestResult:
    """Aggregated backtest results across multiple years."""

    config_name: str
    model_name: str
    features: list[str]
    year_results: list[YearResult]
    all_probs: pd.DataFrame  # columns: y_true, prob, year
    auc: float = 0.0
    top1: int = 0
    top1_pct: float = 0.0
    top3: int = 0
    top3_pct: float = 0.0
    avg_rank: float = 0.0
    n_years: int = 0


@dataclass
class PredictionResult:
    """Predictions for a single target year."""

    year: int
    config_name: str
    model_name: str
    features: list[str]
    predictions: pd.DataFrame  # columns: film, prob, (+ original data columns)


def compute_backtest_metrics(
    year_results: list[YearResult],
    all_probs: pd.DataFrame,
) -> dict:
    """Compute aggregate metrics from backtest year results.

    Returns dict with keys: auc, top1, top1_pct, top3, top3_pct, avg_rank, n_years.
    """
    if not year_results or all_probs.empty:
        return {
            "auc": 0.0, "top1": 0, "top1_pct": 0.0,
            "top3": 0, "top3_pct": 0.0, "avg_rank": 0.0, "n_years": 0,
        }

    from sklearn.metrics import roc_auc_score

    n_years = len(year_results)
    top1 = sum(1 for yr in year_results if yr.correct)
    top3 = sum(1 for yr in year_results if yr.winner_rank <= 3)
    avg_rank = sum(yr.winner_rank for yr in year_results) / n_years

    try:
        auc = roc_auc_score(all_probs["y_true"], all_probs["prob"])
    except ValueError:
        auc = 0.0

    return {
        "auc": auc,
        "top1": top1,
        "top1_pct": top1 / n_years * 100,
        "top3": top3,
        "top3_pct": top3 / n_years * 100,
        "avg_rank": avg_rank,
        "n_years": n_years,
    }
