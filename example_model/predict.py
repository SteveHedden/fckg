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
  data/reports/<category>/shap_values_<year>.csv
    Columns: nominee_label, <feature1>, <feature2>, ...
  data/reports/<category>/shap_summary_<year>.png
    Beeswarm plot of SHAP values across all nominees
  data/reports/<category>/shap/<nominee>_<year>.png
    Individual waterfall plot for each nominee

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
  pip install rdflib scikit-learn pandas numpy python-dotenv matplotlib shap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features.category_extractor import CategoryExtractor
from model.category_model import MODEL_TYPES, OscarCategoryModel, _make_estimator
from model.visualize import (
    compute_shap,
    plot_shap_summary,
    plot_shap_waterfall,
    _safe_stem,
)

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

    # ── 4. Write predictions CSV ───────────────────────────────────────────────
    out_path = output_dir / f"predictions_{predict_year}.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\nPredictions for {args.award_system}/{args.category} ({predict_year}):")
    print(out_df.to_string(index=False))
    print(f"\nWrote: {out_path}")

    # ── 5. SHAP analysis ──────────────────────────────────────────────────────
    print("\nComputing SHAP values ...")
    train_df = data[data["year_film"] < predict_year]
    test_df = data[data["year_film"] == predict_year]

    shap_model_type = model_type if model_type != "LR+RF" else "LR"
    shap_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", _make_estimator(shap_model_type)),
    ])
    X_train = train_df[selected].values
    y_train = train_df["winner"].astype(int).values
    X_test = test_df[selected].values
    shap_pipe.fit(X_train, y_train)

    sv, base, X_test_t = compute_shap(shap_pipe, X_train, X_test, shap_model_type)

    # Build nominee labels aligned with pred_df row order
    test_aligned = data[data["year_film"] == predict_year].copy()
    if extractor.row_type == "nominee":
        labels = (
            test_aligned["candidate_name"].fillna("").astype(str)
            + " ("
            + test_aligned["film"].fillna("").astype(str)
            + ")"
        ).tolist()
    else:
        labels = test_aligned["film"].fillna("").astype(str).tolist()

    # SHAP values CSV
    shap_df = pd.DataFrame(sv, columns=selected)
    shap_df.insert(0, "nominee_label", labels)
    shap_csv = output_dir / f"shap_values_{predict_year}.csv"
    shap_df.to_csv(shap_csv, index=False)
    print(f"  Wrote: {shap_csv.name}")

    # SHAP summary plot
    summary_path = output_dir / f"shap_summary_{predict_year}.png"
    plot_shap_summary(
        sv, X_test_t, selected, summary_path,
        title=f"SHAP Summary — {args.category} ({predict_year})",
    )
    print(f"  Wrote: {summary_path.name}")

    # Per-nominee waterfall plots
    shap_dir = output_dir / "shap"
    shap_dir.mkdir(exist_ok=True)
    for i, label in enumerate(labels):
        film_path = shap_dir / f"{_safe_stem(label)}_{predict_year}.png"
        plot_shap_waterfall(sv[i], base, X_test_t[i], selected, label, film_path)
    print(f"  Wrote: shap/{len(labels)} waterfall plots")


if __name__ == "__main__":
    main()
