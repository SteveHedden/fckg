#!/usr/bin/env python3
"""
feature_selection.py  [Step 3 of 5]
─────────────────────────────────────
Uses Recursive Feature Elimination with Cross-Validation (RFECV) to identify
which features are predictive for a given award category. Run this once per
category before model selection to narrow the feature space.

RFECV fits a logistic regression repeatedly, eliminating the weakest feature
each round, and uses cross-validation to find the optimal subset. It trains
only on historical data (years before the backtest window) so there is no
data leakage into the evaluation period.

Outputs
───────
  data/reports/<category>/selected_features.json  — feature list for step 4
  (also printed to stdout)

Usage
─────
  python feature_selection.py --category best_picture
  python feature_selection.py --category best_actress --award-system oscars
  python feature_selection.py --category best_picture --train-end-year 2020

Supported categories (oscars)
──────────────────────────────
  best_picture, best_director, best_actor, best_actress,
  best_supporting_actor, best_supporting_actress,
  cinematography, editing, score, adapted_screenplay,
  original_screenplay, makeup, costume_design, visual_effects

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
from model.category_model import OscarCategoryModel, _DEFAULT_EXCLUDE

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _candidate_count(df: pd.DataFrame) -> int:
    return len([
        c for c in df.columns
        if c not in _DEFAULT_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: RFECV feature selection for one award category."
    )
    parser.add_argument(
        "--category", required=True,
        help="Category alias (e.g. best_picture, best_actress, cinematography).",
    )
    parser.add_argument(
        "--award-system", default="oscars",
        choices=["oscars", "bafta", "sag"],
        help="Award system (default: oscars).",
    )
    parser.add_argument(
        "--train-end-year", type=int, default=None,
        help="Train RFECV on rows where year_film < this value (default: max year in data).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write selected_features.json (default: data/reports/<category>).",
    )
    args = parser.parse_args()

    # ── 1. Extract features ───────────────────────────────────────────────────
    print(f"Extracting features for {args.award_system}/{args.category} ...")
    try:
        extractor = CategoryExtractor(args.category, award_system=args.award_system)
        data = extractor.extract()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if data.empty:
        print(f"Error: no rows for category '{args.category}'", file=sys.stderr)
        sys.exit(1)

    max_year = int(data["year_film"].max())
    train_end_year = args.train_end_year if args.train_end_year is not None else max_year
    train_df = data[data["year_film"] < train_end_year].copy()

    print(f"  {len(data)} total rows  |  {_candidate_count(train_df)} candidate features")
    print(f"  Training on {len(train_df)} rows (year_film < {train_end_year})")

    if len(train_df) < 30:
        print(f"Error: only {len(train_df)} training rows — need at least 30.", file=sys.stderr)
        sys.exit(1)

    # ── 2. Run RFECV ──────────────────────────────────────────────────────────
    print("Running RFECV (this may take a minute) ...")
    try:
        model = OscarCategoryModel(
            train_df,
            category_name=args.category,
            features="RFECV",
            model_type="LR",
            label_cols=extractor.label_cols,
        )
    except ValueError as exc:
        print(f"Error: RFECV failed: {exc}", file=sys.stderr)
        sys.exit(1)

    selected = model.features
    print(f"\nSelected {len(selected)} features (from {_candidate_count(train_df)} candidates):")
    for i, f in enumerate(selected, 1):
        print(f"  {i:>2}. {f}")

    # ── 3. Write output ────────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.award_system == "oscars":
        output_dir = PROJECT_ROOT / "data" / "reports" / args.category
    else:
        output_dir = PROJECT_ROOT / "data" / "reports" / args.award_system / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "category": args.category,
        "award_system": args.award_system,
        "train_end_year": train_end_year,
        "n_candidates": _candidate_count(train_df),
        "selected_features": selected,
    }
    out_path = output_dir / "selected_features.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}")
    print("Next step: run model_selection.py --category", args.category)


if __name__ == "__main__":
    main()
