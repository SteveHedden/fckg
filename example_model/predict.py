#!/usr/bin/env python3
"""
predict.py  [Step 5 of 5]
──────────────────────────
Trains the selected model on all available historical data and generates
win-probability predictions for the current award season's nominees.

Reads the winning model type and selected features from step 4 outputs
(grid_results.csv and selected_features.json), or accepts them directly
via command-line flags. Predictions are written to a CSV ranked by
estimated win probability.

Output
──────
  data/reports/<category>/predictions_<year>.csv
    Columns: candidate_name (acting categories), film, win_pct, rank

Usage
─────
  # Use step 3/4 outputs automatically
  python predict.py --category best_picture --year 2025

  # Specify model and features explicitly
  python predict.py --category best_picture --year 2025 --model LR
  python predict.py --category best_picture --year 2025 \\
      --model LR --features pga_winner dga_winner sag_winner globe_drama_winner

Requirements
────────────
  pip install rdflib scikit-learn pandas numpy python-dotenv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features.category_extractor import CategoryExtractor
from model.category_model import MODEL_TYPES, OscarCategoryModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_output_dir(category: str, award_system: str, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    if award_system == "oscars":
        return PROJECT_ROOT / "data" / "reports" / category
    return PROJECT_ROOT / "data" / "reports" / award_system / category


def _load_step4_outputs(output_dir: Path) -> tuple[list[str] | None, str | None]:
    """Read selected_features.json and grid_results.csv from step 3/4."""
    features = None
    model_type = None

    features_path = output_dir / "selected_features.json"
    if features_path.exists():
        features = json.loads(features_path.read_text())["selected_features"]

    grid_path = output_dir / "grid_results.csv"
    if grid_path.exists():
        grid_df = pd.read_csv(grid_path)
        if not grid_df.empty:
            model_type = str(grid_df.sort_values("auc", ascending=False).iloc[0]["model"])

    return features, model_type


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5: predict win probabilities for current season nominees."
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
        "--year", type=int, default=None,
        help="Award year to predict (default: most recent year in data).",
    )
    parser.add_argument(
        "--model", default=None, choices=MODEL_TYPES,
        help="Model type to use (default: winning model from grid_results.csv).",
    )
    parser.add_argument(
        "--features", nargs="+", default=None, metavar="FEATURE",
        help="Feature list (default: from selected_features.json).",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = _resolve_output_dir(args.category, args.award_system, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract features ───────────────────────────────────────────────────
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

    predict_year = args.year if args.year is not None else int(data["year_film"].max())
    print(f"  Predicting year: {predict_year}")

    # ── 2. Resolve model type and features ────────────────────────────────────
    saved_features, saved_model = _load_step4_outputs(output_dir)

    selected = args.features or saved_features
    model_type = args.model or saved_model

    if not selected:
        print(
            "Error: no features found. Run feature_selection.py first, "
            "or pass --features explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not model_type:
        print(
            "Error: no model type found. Run model_selection.py first, "
            "or pass --model explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = [f for f in selected if f not in data.columns]
    if missing:
        print(f"Error: features not in data: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"  Model: {model_type}  |  Features: {len(selected)}")

    # ── 3. Train on all history and predict ───────────────────────────────────
    try:
        model = OscarCategoryModel(
            data,
            category_name=args.category,
            features=selected,
            model_type=model_type,
            label_cols=extractor.label_cols,
        )
        result = model.predict(predict_year)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    pred_df = result.predictions.reset_index(drop=True)
    pred_df["rank"] = pred_df.index + 1
    total = pred_df["prob"].sum()
    pred_df["win_pct"] = (pred_df["prob"] / total * 100).round(1) if total > 0 else 0.0

    if extractor.row_type == "nominee":
        out_df = pred_df[["candidate_name", "film", "win_pct", "rank"]].copy()
    else:
        out_df = pred_df[["film", "win_pct", "rank"]].copy()

    # ── 4. Write output ────────────────────────────────────────────────────────
    out_path = output_dir / f"predictions_{predict_year}.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\nPredictions for {args.award_system}/{args.category} ({predict_year}):")
    print(out_df.to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
