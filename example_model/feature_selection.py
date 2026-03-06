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

Flags
─────
  --no-conceptual
                 Disable concept-aware feature reduction before RFECV.
                 By default, conceptual reduction is enabled: within each
                 conceptual group (financial scale, critical reception,
                 audience rating, etc.) only the single best-performing
                 feature is kept, and precursor nominee/winner pairs are
                 reduced to the winner column.

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
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features.category_extractor import CategoryExtractor
from features.feature_groups import reduce_to_group_representatives
from features.leakage import LeakageValidationError
from model.category_model import (
    OscarCategoryModel,
    _DEFAULT_EXCLUDE,
    infer_category_specific_precursor_features,
)

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
    parser.add_argument(
        "--max-features", type=int, default=None,
        metavar="N",
        help=(
            "Hard cap on the number of selected features. "
            "Default: auto-computed as max(3, n_ceremonies // 5) — "
            "the EPV≈5 rule (5 training observations per parameter). "
            "Override with a specific integer to force a tighter or looser limit."
        ),
    )
    parser.add_argument(
        "--no-conceptual", action="store_true",
        help=(
            "Disable concept-aware feature reduction before RFECV. "
            "By default, keeps one representative per conceptual group "
            "(financial, critical reception, etc.) and collapses precursor "
            "nominee/winner pairs to the winner column."
        ),
    )
    parser.add_argument(
        "--check-leakage",
        action="store_true",
        help=(
            "Validate features against system-aware leakage guardrails "
            "before model training."
        ),
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

    n_candidates = _candidate_count(train_df)
    print(f"  {len(data)} total rows  |  {n_candidates} candidate features")
    print(f"  Training on {len(train_df)} rows (year_film < {train_end_year})")

    if len(train_df) < 30:
        print(f"Error: only {len(train_df)} training rows — need at least 30.", file=sys.stderr)
        sys.exit(1)

    # ── 2. Conceptual reduction (optional) ────────────────────────────────────
    candidate_cols = [
        c for c in train_df.columns
        if c not in _DEFAULT_EXCLUDE and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    forced_category_precursors: list[str] = []

    # For SAG categories, Oscar winner columns are future leakage — the Oscars
    # happen weeks after the SAG ceremony, so the winner cannot be known at
    # prediction time. Oscar nominee columns are kept (nominations are announced
    # ~6 weeks before the ceremony, before SAG).
    if args.award_system == "sag":
        leaky = [c for c in candidate_cols
                 if c.startswith("oscars_") and c.endswith("_winner")]
        if leaky:
            print(f"  Excluding {len(leaky)} Oscar winner feature(s) "
                  f"(future leakage for SAG): {leaky}")
            candidate_cols = [c for c in candidate_cols if c not in leaky]

    forced_category_precursors = infer_category_specific_precursor_features(
        args.category, candidate_cols
    )
    if forced_category_precursors:
        print(
            f"  Forcing {len(forced_category_precursors)} category-specific "
            f"precursor feature(s): {forced_category_precursors}"
        )

    if not args.no_conceptual:
        print("\nApplying conceptual feature grouping ...")
        report = reduce_to_group_representatives(
            train_df, candidate_cols, category=args.category
        )
        report.print_summary()
        rfecv_cols = report.selected
        rfecv_df = train_df[list({*rfecv_cols, "winner", "year_film"})].copy()
    else:
        rfecv_df = train_df
        rfecv_cols = candidate_cols   # RFECV will see all candidates

    if forced_category_precursors:
        missing_forced = [c for c in forced_category_precursors if c not in rfecv_cols]
        if missing_forced:
            print(
                f"  Restoring {len(missing_forced)} forced precursor feature(s) "
                f"after conceptual grouping: {missing_forced}"
            )
        rfecv_cols = list(dict.fromkeys([*rfecv_cols, *forced_category_precursors]))
        rfecv_df = train_df[list({*rfecv_cols, "winner", "year_film"})].copy()

    # ── 3. Three-tier precursor handling ─────────────────────────────────────
    #
    # Tier 1 — Category-specific precursors (from infer_category_specific_
    #          precursor_features): always forced, bypass RFECV.
    # Tier 2 — Generic *_winner flags NOT in Tier 1: go through RFECV so the
    #          model decides whether they add value.  Generic *_nominee flags
    #          are excluded entirely (weak/noisy signal).
    # Tier 3 — Everything else: goes through RFECV as normal.
    #
    # RFECV pool = Tier 2 + Tier 3.
    tier1_feats = [c for c in forced_category_precursors if c in rfecv_cols]
    tier1_set = set(tier1_feats)

    generic_nominee_excluded = [
        c for c in rfecv_cols
        if c.endswith("_nominee") and c not in tier1_set
    ]
    generic_nominee_set = set(generic_nominee_excluded)

    tier2_feats = [
        c for c in rfecv_cols
        if c.endswith("_winner") and c not in tier1_set
    ]

    tier3_feats = [
        c for c in rfecv_cols
        if c not in tier1_set
        and c not in generic_nominee_set
        and not (c.endswith("_winner") and c not in tier1_set)
    ]

    rfecv_pool = tier2_feats + tier3_feats

    if tier1_feats:
        print(f"\n  Tier 1 (forced category-specific, bypass RFECV): {tier1_feats}")
    if tier2_feats:
        print(f"  Tier 2 (generic winners, go through RFECV): {tier2_feats}")
    if generic_nominee_excluded:
        print(f"  Excluded generic nominees (noisy): {generic_nominee_excluded}")

    print(f"\nRunning RFECV on {len(rfecv_pool)} features "
          f"(Tier 2 + Tier 3; forcing {len(tier1_feats)} Tier 1 features) ...")

    if rfecv_pool:
        rfecv_pool_df = train_df[
            list({*rfecv_pool, "winner", "year_film"})
        ].copy()
        try:
            model = OscarCategoryModel(
                rfecv_pool_df,
                category_name=args.category,
                features="RFECV",
                model_type="LR",
                label_cols=extractor.label_cols,
                award_system=args.award_system,
                check_leakage=args.check_leakage,
            )
            rfecv_selected = model.features
        except LeakageValidationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except ValueError as exc:
            print(f"Warning: RFECV failed ({exc}); keeping all pool features.",
                  file=sys.stderr)
            rfecv_selected = rfecv_pool
    else:
        rfecv_selected = []

    # ── 4. Apply EPV-based feature cap ────────────────────────────────────────
    # Tier 1 (forced) features are always kept in full.
    # The EPV cap (n_ceremonies ÷ 5) applies only to RFECV-selected features.
    n_ceremonies = int(train_df["year_film"].nunique())
    other_cap = args.max_features if args.max_features is not None else max(3, n_ceremonies // 5)

    other_feats = rfecv_selected

    print(f"\nFeature cap (EPV≈5, n_ceremonies={n_ceremonies}):")
    print(f"  Tier 1 features (always kept, {len(tier1_feats)}): {tier1_feats}")

    if len(other_feats) > other_cap:
        source = (
            f"--max-features={other_cap}"
            if args.max_features is not None
            else f"n_ceremonies={n_ceremonies} ÷ 5 = {other_cap}"
        )
        print(f"  RFECV-selected features: {len(other_feats)} → capping at {other_cap} ({source})")
        prep = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_rfe = prep.fit_transform(rfecv_df[rfecv_pool].values)
        y_rfe = rfecv_df["winner"].astype(int).values
        rfe = RFE(
            LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
            n_features_to_select=other_cap,
        )
        rfe.fit(X_rfe, y_rfe)
        other_feats = [rfecv_pool[i] for i in range(len(rfecv_pool)) if rfe.support_[i]]
        print(f"  → {len(other_feats)} features after RFE cap")
    else:
        print(f"  RFECV-selected features: {len(other_feats)} — within limit of {other_cap}.")

    selected = tier1_feats + other_feats

    if args.check_leakage:
        try:
            OscarCategoryModel(
                train_df[list({*selected, "winner", "year_film"})].copy(),
                category_name=args.category,
                features=selected,
                model_type="LR",
                label_cols=extractor.label_cols,
                award_system=args.award_system,
                check_leakage=True,
            )
        except LeakageValidationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    # Build tier lookup for display
    tier2_set = set(tier2_feats)
    rfecv_survived_set = set(other_feats)

    print(f"\nFinal {len(selected)} features (from {_candidate_count(train_df)} candidates):")
    for i, f in enumerate(selected, 1):
        if f in tier1_set:
            tag = "[T1 forced]"
        elif f in tier2_set:
            tag = "[T2 generic winner]"
        else:
            tag = "[T3]"
        print(f"  {i:>2}. {f}  {tag}")

    # ── 5. Write output ────────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "data" / "reports" / args.award_system / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "category": args.category,
        "award_system": args.award_system,
        "train_end_year": train_end_year,
        "n_candidates": n_candidates,
        "n_ceremonies": n_ceremonies,
        "feature_cap": other_cap,
        "conceptual_reduction": not args.no_conceptual,
        "tier1_forced_precursors": tier1_feats,
        "generic_precursor_nominees_excluded": generic_nominee_excluded,
        "selected_features": selected,
    }
    out_path = output_dir / "selected_features.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}")
    print("Next step: run model_selection.py --category", args.category)


if __name__ == "__main__":
    main()
