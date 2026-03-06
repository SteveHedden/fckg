"""CLI entrypoint for the config-first FCKG prediction pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.leakage import LeakageValidationError
from features.category_extractor import CategoryExtractor as _DefaultCategoryExtractor
from model.category_model import (
    OscarCategoryModel,
    _DEFAULT_EXCLUDE,
    infer_category_specific_precursor_features,
)

from fckg.config import ConfigValidationError, PipelineConfig
from fckg.feature_groups import reduce_to_group_representatives

try:
    # Prefer the legacy extractor when present so config-first RFECV policy
    # matches March-3 report generation behavior.
    from example_model.features.category_extractor import CategoryExtractor as _LegacyCategoryExtractor
except ImportError:
    _LegacyCategoryExtractor = None

CategoryExtractor = _LegacyCategoryExtractor or _DefaultCategoryExtractor


def _nominee_labels(df: pd.DataFrame, row_type: str) -> pd.Series:
    if row_type == "film":
        return df["film"].fillna("(unknown)").astype(str)
    names = df["candidate_name"].fillna("(unknown)").astype(str)
    films = df["film"].fillna("(unknown)").astype(str)
    return names + " (" + films + ")"


def _resolve_selected_features(
    config: PipelineConfig,
    data: pd.DataFrame,
    *,
    check_leakage: bool,
) -> tuple[list[str], str, dict[str, object]]:
    def _candidate_count(df: pd.DataFrame) -> int:
        return len(
            [
                c for c in df.columns
                if c not in _DEFAULT_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
            ]
        )

    def _resolve_rfecv_with_legacy_policy(
        train_slice: pd.DataFrame,
    ) -> tuple[list[str], dict[str, object]]:
        candidate_cols = [
            c
            for c in train_slice.columns
            if c not in _DEFAULT_EXCLUDE and pd.api.types.is_numeric_dtype(train_slice[c])
        ]
        n_candidates = _candidate_count(train_slice)

        if config.award_system == "sag":
            candidate_cols = [
                c
                for c in candidate_cols
                if not (c.startswith("oscars_") and c.endswith("_winner"))
            ]

        forced_category_precursors = infer_category_specific_precursor_features(
            config.category, candidate_cols
        )

        # Keep parity with legacy step-3 behavior: conceptual grouping is enabled.
        report = reduce_to_group_representatives(
            train_slice,
            candidate_cols,
            category=config.category,
        )
        rfecv_cols = report.selected
        rfecv_df = train_slice[list({*rfecv_cols, "winner", "year_film"})].copy()

        if forced_category_precursors:
            missing_forced = [c for c in forced_category_precursors if c not in rfecv_cols]
            if missing_forced:
                rfecv_cols = list(dict.fromkeys([*rfecv_cols, *missing_forced]))
                rfecv_df = train_slice[list({*rfecv_cols, "winner", "year_film"})].copy()

        tier1_feats = [c for c in forced_category_precursors if c in rfecv_cols]
        tier1_set = set(tier1_feats)
        generic_nominee_excluded = [
            c for c in rfecv_cols if c.endswith("_nominee") and c not in tier1_set
        ]
        generic_nominee_set = set(generic_nominee_excluded)
        tier2_feats = [
            c for c in rfecv_cols if c.endswith("_winner") and c not in tier1_set
        ]
        tier3_feats = [
            c
            for c in rfecv_cols
            if c not in tier1_set
            and c not in generic_nominee_set
            and not (c.endswith("_winner") and c not in tier1_set)
        ]
        rfecv_pool = tier2_feats + tier3_feats

        if rfecv_pool:
            rfecv_pool_df = train_slice[list({*rfecv_pool, "winner", "year_film"})].copy()
            try:
                model = OscarCategoryModel(
                    rfecv_pool_df,
                    category_name=config.category,
                    features="RFECV",
                    model_type="LR",
                    label_cols=config.label_cols,
                    award_system=config.award_system,
                    check_leakage=check_leakage,
                )
                rfecv_selected = model.features
            except ValueError:
                rfecv_selected = rfecv_pool
        else:
            rfecv_selected = []

        n_ceremonies = int(train_slice["year_film"].nunique())
        other_cap = max(3, n_ceremonies // 5)
        other_feats = rfecv_selected
        if len(other_feats) > other_cap and rfecv_pool:
            prep = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            X_rfe = prep.fit_transform(rfecv_df[rfecv_pool].values)
            y_rfe = rfecv_df["winner"].astype(int).values
            rfe = RFE(
                LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
                n_features_to_select=other_cap,
            )
            rfe.fit(X_rfe, y_rfe)
            other_feats = [rfecv_pool[i] for i in range(len(rfecv_pool)) if rfe.support_[i]]

        selected = tier1_feats + other_feats

        if check_leakage and selected:
            OscarCategoryModel(
                train_slice[list({*selected, "winner", "year_film"})].copy(),
                category_name=config.category,
                features=selected,
                model_type="LR",
                label_cols=config.label_cols,
                award_system=config.award_system,
                check_leakage=True,
            )

        metadata = {
            "train_end_year": config.ceremony_year,
            "n_candidates": n_candidates,
            "n_ceremonies": n_ceremonies,
            "feature_cap": other_cap,
            "conceptual_reduction": True,
            "tier1_forced_precursors": tier1_feats,
            "generic_precursor_nominees_excluded": generic_nominee_excluded,
        }
        return selected, metadata

    feature_spec = config.features
    if isinstance(feature_spec, list):
        missing = [f for f in feature_spec if f not in data.columns]
        if missing:
            raise ValueError(
                f"Explicit feature list includes unknown columns: {missing}"
            )
        return list(feature_spec), "explicit", {}

    if feature_spec == "RFECV":
        train_slice = data[data["year_film"] < config.ceremony_year].copy()
        if len(train_slice) < 30:
            raise ValueError(
                "RFECV requires at least 30 training rows where "
                f"year_film < ceremony_year ({config.ceremony_year}); found {len(train_slice)}"
            )
        selected, metadata = _resolve_rfecv_with_legacy_policy(train_slice)
        return selected, "RFECV", metadata

    if feature_spec == "ALL":
        model = OscarCategoryModel(
            data,
            category_name=config.category,
            features="ALL",
            model_type="LR",
            label_cols=config.label_cols,
            award_system=config.award_system,
            check_leakage=check_leakage,
        )
        return model.features, "ALL", {}

    raise ValueError(f"Unsupported features spec: {feature_spec!r}")


def _write_selected_features(
    output_dir: Path,
    *,
    config: PipelineConfig,
    selected_features: list[str],
    feature_source: str,
    metadata: dict[str, object] | None = None,
) -> None:
    payload = {
        "category": config.category,
        "award_system": config.award_system,
        "ceremony_year": config.ceremony_year,
        "feature_mode": feature_source,
        "selected_features": selected_features,
    }
    if metadata:
        payload.update(metadata)
    (output_dir / "selected_features.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _write_backtest_outputs(
    output_dir: Path,
    *,
    model: OscarCategoryModel,
    data: pd.DataFrame,
    row_type: str,
    start_year: int,
    end_year: int,
    selected_features: list[str],
) -> None:
    result = model.backtest(start_year=start_year, end_year=end_year)
    grid_df = pd.DataFrame(
        [
            {
                "model": model.model_type,
                "auc": float(result.auc),
                "top1": int(result.top1),
                "top1_pct": float(result.top1_pct),
                "top3": int(result.top3),
                "top3_pct": float(result.top3_pct),
                "avg_rank": float(result.avg_rank),
                "n_years": int(result.n_years),
                "n_features": len(selected_features),
            }
        ]
    )
    grid_df.to_csv(output_dir / "grid_results.csv", index=False)

    rows: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        train_rows = data[data["year_film"] < year]
        year_rows = data[data["year_film"] == year]
        if len(train_rows) < 30 or year_rows.empty:
            continue
        if int(year_rows["winner"].astype(int).sum()) == 0:
            continue

        try:
            pred = model.predict(year)
        except ValueError:
            continue

        pred_df = pred.predictions.reset_index(drop=True)
        if pred_df.empty:
            continue
        pred_df["rank"] = pred_df.index + 1
        pred_df["nominee_label"] = _nominee_labels(pred_df, row_type=row_type)
        winner_rows = pred_df[pred_df["winner"] == True]
        if winner_rows.empty:
            continue

        top_pick = str(pred_df.iloc[0]["nominee_label"])
        actual_winner = str(winner_rows.iloc[0]["nominee_label"])
        winner_rank = int(winner_rows.iloc[0]["rank"])
        correct = bool(winner_rank == 1)

        for _, row in pred_df.iterrows():
            rows.append(
                {
                    "year_film": int(year),
                    "nominee_label": str(row["nominee_label"]),
                    "winner": bool(row["winner"]),
                    "prob": round(float(row["prob"]), 4),
                    "rank": int(row["rank"]),
                    "top_pick": top_pick,
                    "actual_winner": actual_winner,
                    "winner_rank": winner_rank,
                    "correct": correct,
                }
            )

    backcast_columns = [
        "year_film",
        "nominee_label",
        "winner",
        "prob",
        "rank",
        "top_pick",
        "actual_winner",
        "winner_rank",
        "correct",
    ]
    backcast_df = pd.DataFrame(rows, columns=backcast_columns)
    backcast_df.to_csv(output_dir / "backcast_results.csv", index=False)


def _compute_legacy_feature_importance(
    *,
    data: pd.DataFrame,
    selected_features: list[str],
    model_type: str,
    category: str,
    ceremony_year: int,
) -> pd.DataFrame:
    """Match March-3 feature importance behavior (signed linear coefficients)."""
    from model.category_model import _make_estimator

    train_df = data[data["year_film"] < ceremony_year].copy()
    if train_df.empty:
        return pd.DataFrame(columns=["rank", "feature", "importance"])

    fit_model_type = "LR" if model_type == "LR+RF" else model_type
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                _make_estimator(
                    fit_model_type,
                    category_name=category,
                    features=selected_features,
                ),
            ),
        ]
    )
    X_train = train_df[selected_features].values
    y_train = train_df["winner"].astype(int).values

    if fit_model_type == "GB":
        n_winners = y_train.sum()
        n_total = len(y_train)
        weight_ratio = (n_total - n_winners) / max(n_winners, 1)
        sample_weight = np.where(y_train == 1, weight_ratio, 1.0)
        pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        pipe.fit(X_train, y_train)

    if fit_model_type in ("LR", "Ridge", "ConstrainedLR"):
        raw_imp = pipe["model"].coef_[0]
    else:
        raw_imp = pipe["model"].feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": selected_features,
            "importance": raw_imp,
        }
    )
    importance_df = importance_df.reindex(
        pd.Series(np.abs(raw_imp)).sort_values(ascending=False).index
    ).reset_index(drop=True)
    importance_df.insert(0, "rank", importance_df.index + 1)
    return importance_df[["rank", "feature", "importance"]]


def _write_optional_shap_outputs(
    output_dir: Path,
    *,
    data: pd.DataFrame,
    selected_features: list[str],
    model_type: str,
    category: str,
    ceremony_year: int,
    row_type: str,
) -> None:
    from model.category_model import _make_estimator
    from model.visualize import (
        _safe_stem,
        compute_shap,
        plot_shap_summary,
        plot_shap_waterfall,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train_df = data[data["year_film"] < ceremony_year].copy()
    test_df = data[data["year_film"] == ceremony_year].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"SHAP requested, but insufficient data for ceremony_year={ceremony_year}"
        )

    shap_model_type = "LR" if model_type == "LR+RF" else model_type
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                _make_estimator(
                    shap_model_type,
                    category_name=category,
                    features=selected_features,
                ),
            ),
        ]
    )
    X_train = train_df[selected_features].values
    y_train = train_df["winner"].astype(int).values
    X_test = test_df[selected_features].values
    pipe.fit(X_train, y_train)

    sv, base, X_test_t = compute_shap(pipe, X_train, X_test, shap_model_type)

    labels = _nominee_labels(test_df, row_type=row_type).tolist()
    shap_df = pd.DataFrame(sv, columns=selected_features)
    shap_df.insert(0, "nominee_label", labels)
    shap_df.to_csv(output_dir / f"shap_values_{ceremony_year}.csv", index=False)

    plot_shap_summary(
        sv,
        X_test_t,
        selected_features,
        output_dir / f"shap_summary_{ceremony_year}.png",
        title=f"SHAP Summary — {category} ({ceremony_year})",
    )

    shap_dir = output_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    for i, label in enumerate(labels):
        out_path = shap_dir / f"{_safe_stem(label)}_{ceremony_year}.png"
        plot_shap_waterfall(sv[i], base, X_test_t[i], selected_features, label, out_path)


def _cleanup_optional_artifacts(output_dir: Path) -> None:
    # Backtest-managed artifacts
    for name in ("grid_results.csv", "backcast_results.csv", "backcast_summary.txt"):
        path = output_dir / name
        if path.exists():
            path.unlink()

    # SHAP-managed artifacts
    for pattern in ("shap_values_*.csv", "shap_summary_*.png", "shap_waterfall_*.png"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    shap_dir = output_dir / "shap"
    if shap_dir.exists():
        if shap_dir.is_dir():
            shutil.rmtree(shap_dir)
        else:
            shap_dir.unlink()


def run_pipeline(config_path: str | Path, *, check_leakage: bool = False) -> Path:
    config = PipelineConfig.from_yaml(config_path)
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_optional_artifacts(output_dir)

    extractor = CategoryExtractor(config.category, award_system=config.award_system)
    data = extractor.extract()
    if data.empty:
        raise ValueError(
            f"No rows extracted for {config.award_system}/{config.category}; cannot run pipeline"
        )

    selected_features, feature_source, feature_metadata = _resolve_selected_features(
        config,
        data,
        check_leakage=check_leakage,
    )
    if not selected_features:
        raise ValueError("No features were selected for training")

    _write_selected_features(
        output_dir,
        config=config,
        selected_features=selected_features,
        feature_source=feature_source,
        metadata=feature_metadata,
    )

    try:
        model = OscarCategoryModel(
            data,
            category_name=config.category,
            features=selected_features,
            model_type=config.model.type,
            label_cols=config.label_cols,
            award_system=config.award_system,
            check_leakage=check_leakage,
            estimator_params=config.model.hyperparameters or None,
        )
    except LeakageValidationError as exc:
        raise LeakageValidationError(
            f"Leakage validation failed for {config.award_system}/{config.category} "
            f"(ceremony_year={config.ceremony_year}): {exc}"
        ) from exc

    if config.backtest is not None:
        _write_backtest_outputs(
            output_dir,
            model=model,
            data=data,
            row_type=extractor.row_type,
            start_year=config.backtest.start_year,
            end_year=config.backtest.end_year,
            selected_features=selected_features,
        )

    prediction = model.predict(config.ceremony_year)
    pred_df = prediction.predictions.reset_index(drop=True)
    pred_df["rank"] = pred_df.index + 1

    raw = np.clip(pred_df["prob"].astype(float).values, 1e-7, 1 - 1e-7)
    logits = np.log(raw / (1 - raw))
    exp_logits = np.exp(logits - logits.max())
    pred_df["win_pct"] = (exp_logits / exp_logits.sum() * 100).round(1)

    if extractor.row_type == "nominee":
        out_df = pred_df[["candidate_name", "film", "win_pct", "rank"]].copy()
    else:
        out_df = pred_df[["film", "win_pct", "rank"]].copy()
    out_df.to_csv(output_dir / f"predictions_{config.ceremony_year}.csv", index=False)

    importance_df = _compute_legacy_feature_importance(
        data=data,
        selected_features=selected_features,
        model_type=config.model.type,
        category=config.category,
        ceremony_year=config.ceremony_year,
    )
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    if config.shap:
        _write_optional_shap_outputs(
            output_dir,
            data=data,
            selected_features=selected_features,
            model_type=config.model.type,
            category=config.category,
            ceremony_year=config.ceremony_year,
            row_type=extractor.row_type,
        )

    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_payload = config.to_resolved_dict(
        selected_features=selected_features,
        check_leakage=check_leakage,
    )
    resolved_payload["artifacts"] = {
        "resolved_config": "resolved_config.yaml",
        "selected_features": "selected_features.json",
        "predictions": f"predictions_{config.ceremony_year}.csv",
        "feature_importance": "feature_importance.csv",
        "grid_results": "grid_results.csv" if config.backtest else None,
        "backcast_results": "backcast_results.csv" if config.backtest else None,
    }
    resolved_config_path.write_text(
        yaml.safe_dump(resolved_payload, sort_keys=False),
        encoding="utf-8",
    )

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fckg",
        description="Film Club Knowledge Graph command-line interface.",
    )
    parser.add_argument("--version", action="version", version="fckg 0.1.0")

    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser(
        "run",
        help="Run config-first extraction/training/backtest/prediction pipeline",
    )
    run_parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    run_parser.add_argument(
        "--check-leakage",
        action="store_true",
        help="Enable temporal leakage validation during feature resolution and model runs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 1

    try:
        output_dir = run_pipeline(args.config, check_leakage=bool(args.check_leakage))
    except (ConfigValidationError, LeakageValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
