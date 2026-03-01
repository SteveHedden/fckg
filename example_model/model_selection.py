#!/usr/bin/env python3
"""
model_selection.py  [Step 4 of 5]
───────────────────────────────────
Backtests four model types (Logistic Regression, Random Forest, Gradient
Boosting, LR+RF ensemble) using the features selected in step 3, then picks
the winner by AUC and writes full backcast results.

Reads selected_features.json written by feature_selection.py, or accepts an
explicit feature list via --features if you want to skip step 3 or override it.

Outputs (written to data/reports/<category>/)
─────────────────────────────────────────────
  grid_results.csv        — AUC, top-1%, top-3%, avg-rank for every model type
  backcast_results.csv    — year-by-year nominee probabilities and ranks
  backcast_summary.txt    — summary stats for the winning model
  feature_importance.csv  — feature importance ranking for the winning model

Usage
─────
  # Use features from step 3
  python model_selection.py --category best_picture

  # Explicit feature list (skips step 3 output)
  python model_selection.py --category best_picture --features pga_winner dga_winner sag_winner

  # Specify backtest window
  python model_selection.py --category best_picture --start-year 2010 --end-year 2024

  # Evaluate only specific model types
  python model_selection.py --category best_picture --model-types LR RF

Requirements
────────────
  pip install rdflib scikit-learn pandas numpy python-dotenv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features.category_extractor import CategoryExtractor
from model.category_model import MODEL_TYPES, OscarCategoryModel, _make_estimator
from model.evaluation import YearResult, compute_backtest_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _format_pct(v: float) -> str:
    return f"{v:.1f}%"


def _nominee_labels(df: pd.DataFrame, row_type: str) -> pd.Series:
    if row_type == "film":
        return df["film"].fillna("(unknown)").astype(str)
    return (
        df["candidate_name"].fillna("(unknown)").astype(str)
        + " ("
        + df["film"].fillna("(unknown)").astype(str)
        + ")"
    )


def _fit_pipeline(model_type: str, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", _make_estimator(model_type)),
    ])
    if model_type == "GB":
        n_winners = int(y_train.sum())
        weight_ratio = (len(y_train) - n_winners) / max(n_winners, 1)
        pipe.fit(X_train, y_train, model__sample_weight=np.where(y_train == 1, weight_ratio, 1.0))
    else:
        pipe.fit(X_train, y_train)
    return pipe


def _load_selected_features(category: str, award_system: str, output_dir: Path) -> list[str] | None:
    path = output_dir / "selected_features.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("selected_features")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: grid search across model types and backtest evaluation."
    )
    parser.add_argument(
        "--category", required=True,
        help="Category alias (e.g. best_picture, best_actress, cinematography).",
    )
    parser.add_argument(
        "--award-system", default="oscars",
        choices=["oscars", "bafta", "sag"],
    )
    parser.add_argument(
        "--features", nargs="+", default=None, metavar="FEATURE",
        help="Explicit feature list (overrides selected_features.json from step 3).",
    )
    parser.add_argument(
        "--start-year", type=int, default=None,
        help="First year to include in backtest (default: max_year - 9).",
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="Last year to include in backtest (default: max year in data).",
    )
    parser.add_argument(
        "--model-types", nargs="+", choices=MODEL_TYPES, default=list(MODEL_TYPES),
        help=f"Model types to evaluate (default: all — {MODEL_TYPES}).",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    # ── 1. Resolve output directory ───────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.award_system == "oscars":
        output_dir = PROJECT_ROOT / "data" / "reports" / args.category
    else:
        output_dir = PROJECT_ROOT / "data" / "reports" / args.award_system / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Extract features ───────────────────────────────────────────────────
    print(f"Extracting features for {args.award_system}/{args.category} ...")
    try:
        extractor = CategoryExtractor(args.category, award_system=args.award_system)
        data = extractor.extract()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if data.empty:
        print(f"Error: no rows for '{args.category}'", file=sys.stderr)
        sys.exit(1)

    max_year = int(data["year_film"].max())
    start_year = args.start_year if args.start_year is not None else max_year - 9
    end_year = args.end_year if args.end_year is not None else max_year

    # ── 3. Resolve feature list ────────────────────────────────────────────────
    if args.features:
        selected = list(args.features)
        missing = [f for f in selected if f not in data.columns]
        if missing:
            print(f"Error: features not in data: {missing}", file=sys.stderr)
            sys.exit(1)
        print(f"Using {len(selected)} explicitly provided features")
    else:
        selected = _load_selected_features(args.category, args.award_system, output_dir)
        if selected:
            print(f"Loaded {len(selected)} features from selected_features.json")
        else:
            print(
                "Warning: no selected_features.json found. "
                "Run feature_selection.py first, or pass --features explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── 4. Grid search across model types ─────────────────────────────────────
    print(f"\nBacktesting {args.model_types} on years {start_year}–{end_year} ...")
    grid_rows = []
    model_results = {}
    for model_type in args.model_types:
        try:
            model = OscarCategoryModel(
                data,
                category_name=args.category,
                features=selected,
                model_type=model_type,
                label_cols=extractor.label_cols,
            )
            result = model.backtest(start_year=start_year, end_year=end_year)
        except ValueError as exc:
            print(f"  Skipping {model_type}: {exc}", file=sys.stderr)
            continue
        model_results[model_type] = result
        grid_rows.append({
            "model": model_type,
            "auc": float(result.auc),
            "top1": int(result.top1),
            "top1_pct": float(result.top1_pct),
            "top3": int(result.top3),
            "top3_pct": float(result.top3_pct),
            "avg_rank": float(result.avg_rank),
            "n_years": int(result.n_years),
            "n_features": len(selected),
        })

    if not grid_rows:
        print("Error: no model runs produced results", file=sys.stderr)
        sys.exit(1)

    grid_df = pd.DataFrame(grid_rows).sort_values("auc", ascending=False).reset_index(drop=True)
    grid_df.to_csv(output_dir / "grid_results.csv", index=False)

    display = grid_df.copy()
    display["top1_pct"] = display["top1_pct"].map(_format_pct)
    display["top3_pct"] = display["top3_pct"].map(_format_pct)
    print("\nGrid results (sorted by AUC):")
    print(display.to_string(index=False))

    winning_type = str(grid_df.iloc[0]["model"])
    print(f"\nWinning model: {winning_type}")

    # ── 5. Full backcast with winning model ────────────────────────────────────
    winning_model = OscarCategoryModel(
        data,
        category_name=args.category,
        features=selected,
        model_type=winning_type,
        label_cols=extractor.label_cols,
    )

    importance_df = winning_model.feature_importance(target_year=end_year)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    backcast_rows, year_results, all_probs = [], [], []
    for year in range(start_year, end_year + 1):
        if len(data[data["year_film"] < year]) < 30 or data[data["year_film"] == year].empty:
            continue
        if data[data["year_film"] == year]["winner"].astype(int).sum() == 0:
            continue

        pred = winning_model.predict(year)
        pred_df = pred.predictions.reset_index(drop=True)
        pred_df["rank"] = pred_df.index + 1
        pred_df["nominee_label"] = _nominee_labels(pred_df, extractor.row_type)
        winner_rows = pred_df[pred_df["winner"] == True]
        if winner_rows.empty:
            continue

        top_pick = str(pred_df.iloc[0]["nominee_label"])
        actual_winner = str(winner_rows.iloc[0]["nominee_label"])
        winner_rank = int(winner_rows.iloc[0]["rank"])
        correct = winner_rank == 1

        year_results.append(YearResult(
            year=year, top_pick=top_pick, actual_winner=actual_winner,
            winner_rank=winner_rank, winner_prob=float(winner_rows.iloc[0]["prob"]),
            correct=correct,
        ))
        for _, row in pred_df.iterrows():
            backcast_rows.append({
                "year_film": year, "nominee_label": str(row["nominee_label"]),
                "winner": bool(row["winner"]), "prob": round(float(row["prob"]), 4),
                "rank": int(row["rank"]), "top_pick": top_pick,
                "actual_winner": actual_winner, "winner_rank": winner_rank, "correct": correct,
            })
            all_probs.append({"y_true": int(bool(row["winner"])), "prob": float(row["prob"]), "year": year})

    pd.DataFrame(backcast_rows).to_csv(output_dir / "backcast_results.csv", index=False)

    metrics = compute_backtest_metrics(year_results, pd.DataFrame(all_probs))
    summary = "\n".join([
        f"Category: {args.category}", f"Award system: {args.award_system}",
        f"Winning model: {winning_type}", f"Features ({len(selected)}): {selected}",
        f"Years: {start_year}–{end_year}  (n={metrics['n_years']})",
        f"AUC: {metrics['auc']:.4f}",
        f"top1: {metrics['top1']} ({metrics['top1_pct']:.1f}%)",
        f"top3: {metrics['top3']} ({metrics['top3_pct']:.1f}%)",
        f"avg_rank: {metrics['avg_rank']:.4f}",
    ])
    (output_dir / "backcast_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"\n{summary}")
    print(f"\nWrote reports to: {output_dir}")
    print(f"Next step: run predict.py --category {args.category} --model {winning_type}")


if __name__ == "__main__":
    main()
