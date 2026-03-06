"""Generic Oscar category prediction model.

Replaces category-specific trainers (BestPictureModel, BestActorModel) with a
single configurable class that works for any nominee-level or film-level category.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import (
    BacktestResult,
    PredictionResult,
    YearResult,
    compute_backtest_metrics,
)
from features.leakage import (
    LeakageValidationError,
    check_leakage,
    format_leakage_violations,
)

MODEL_TYPES = ["LR", "Ridge", "ConstrainedLR", "RF", "GB", "LR+RF"]

# Category aliases that emit category-specific precursor columns, mapped to the
# shared suffix used in column names: <system>_<prefix>_(nominee|winner)
_CATEGORY_PREFIX_BY_ALIAS: dict[str, str] = {
    "best_actor": "actor",
    "best_actress": "actress",
    "best_supporting_actor": "supporting_actor",
    "best_supporting_actress": "supporting_actress",
    "best_director": "director",
    "best_original_screenplay": "original_screenplay",
    "best_adapted_screenplay": "adapted_screenplay",
    "cinematography": "cinematography",
    "film_editing": "film_editing",
    "original_score": "original_score",
    "original_song": "original_song",
    "production_design": "production_design",
    "costume_design": "costume_design",
    "sound": "sound",
    "visual_effects": "visual_effects",
    "makeup": "makeup",
    "international_film": "international_film",
    "documentary": "documentary",
    "animated_film": "animated_film",
}

_PRECURSOR_SYSTEM_PREFIXES: tuple[str, ...] = (
    "oscars",
    "bafta",
    "sag",
    "golden_globes",
    "pga",
    "dga",
)

# Columns that are identifiers, targets, or free-text — never used as features.
_DEFAULT_EXCLUDE: frozenset[str] = frozenset({
    "film", "film_uri", "year_film", "tmdb_id", "imdb_id",
    "nom", "nominee", "nominee_type", "winner",
    "candidate_id", "candidate_name",
    "release_date", "tagline", "mpaa_rating", "original_language",
    # has_tagline is a data-completeness artifact (missing TMDB records),
    # not a genuine signal — excluded globally.
    "has_tagline",
})


def infer_category_specific_precursor_features(category_name: str, features: list[str]) -> list[str]:
    alias = str(category_name).strip().lower().replace(" ", "_")
    prefix = _CATEGORY_PREFIX_BY_ALIAS.get(alias)
    if not prefix:
        return []

    expected: set[str] = set()
    for system in _PRECURSOR_SYSTEM_PREFIXES:
        expected.add(f"{system}_{prefix}_nominee")
        expected.add(f"{system}_{prefix}_winner")

    return [f for f in features if f in expected]


def constrained_nonnegative_indices(category_name: str, features: list[str]) -> tuple[int, ...]:
    return tuple(
        i for i, f in enumerate(features)
        if f.endswith("_winner") or f.endswith("_nominee")
    )


class ConstrainedLogisticRegression(BaseEstimator, ClassifierMixin):
    """Binary logistic regression with optional non-negative coefficients."""

    def __init__(
        self,
        nonnegative_indices: tuple[int, ...] = (),
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.nonnegative_indices = tuple(nonnegative_indices)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_weights(self, y: np.ndarray, sample_weight: np.ndarray | None) -> np.ndarray:
        if sample_weight is None:
            w = np.ones_like(y, dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float).copy()

        if self.class_weight == "balanced":
            n = len(y)
            n_pos = max(int(y.sum()), 1)
            n_neg = max(int(n - y.sum()), 1)
            pos_w = n / (2.0 * n_pos)
            neg_w = n / (2.0 * n_neg)
            w *= np.where(y == 1, pos_w, neg_w)
        return w

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if len(np.unique(y)) < 2:
            raise ValueError("ConstrainedLR requires both winner and non-winner rows")

        n_features = X.shape[1]
        weights = self._compute_weights(y, sample_weight)
        nonnegative = {i for i in self.nonnegative_indices if 0 <= i < n_features}
        l2 = 1.0 / max(float(self.C), 1e-9)

        def objective(params: np.ndarray) -> float:
            b0 = params[0]
            b = params[1:]
            probs = self._sigmoid(b0 + X @ b)
            eps = 1e-12
            nll = -np.mean(weights * (y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)))
            reg = 0.5 * l2 * float(b @ b)
            return float(nll + reg)

        x0 = np.zeros(n_features + 1, dtype=float)
        bounds = [(None, None)]
        bounds.extend((0.0, None) if i in nonnegative else (None, None) for i in range(n_features))

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(self.max_iter)},
        )
        if not res.success:
            raise ValueError(f"ConstrainedLR optimization failed: {res.message}")

        self.intercept_ = np.array([float(res.x[0])], dtype=float)
        self.coef_ = np.asarray(res.x[1:], dtype=float).reshape(1, -1)
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = n_features
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.intercept_[0] + X @ self.coef_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise ValueError("ConstrainedLR is not fitted")
        p1 = self._sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _make_estimator(
    name: str,
    *,
    category_name: str | None = None,
    features: list[str] | None = None,
    estimator_params: dict[str, object] | None = None,
):
    """Return a fresh sklearn estimator by short name."""
    def validated_kwargs(defaults: dict[str, object], model_name: str) -> dict[str, object]:
        overrides = estimator_params or {}
        unknown = sorted(set(overrides) - set(defaults))
        if unknown:
            allowed = sorted(defaults)
            raise ValueError(
                f"Unsupported hyperparameters for {model_name}: {unknown}. "
                f"Allowed keys: {allowed}"
            )
        merged = dict(defaults)
        merged.update(overrides)
        return merged

    if name == "LR":
        kwargs = validated_kwargs(
            {
                "random_state": 42,
                "max_iter": 1000,
                "class_weight": "balanced",
            },
            "LR",
        )
        return LogisticRegression(**kwargs)
    if name == "Ridge":
        # Stronger L2 regularisation (C=0.1) than plain LR.  Shrinks correlated
        # precursor coefficients proportionally so none flip to spurious negatives.
        # penalty='l2' is the default so omitted to avoid the sklearn 1.8 deprecation.
        kwargs = validated_kwargs(
            {
                "C": 0.1,
                "solver": "lbfgs",
                "random_state": 42,
                "max_iter": 1000,
                "class_weight": "balanced",
            },
            "Ridge",
        )
        return LogisticRegression(**kwargs)
    if name == "ConstrainedLR":
        nonnegative_indices: tuple[int, ...] = ()
        if category_name and features:
            nonnegative_indices = constrained_nonnegative_indices(category_name, features)
        kwargs = validated_kwargs(
            {
                "nonnegative_indices": nonnegative_indices,
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
            },
            "ConstrainedLR",
        )
        return ConstrainedLogisticRegression(**kwargs)
    if name == "RF":
        kwargs = validated_kwargs(
            {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": 42,
            },
            "RF",
        )
        return RandomForestClassifier(**kwargs)
    if name == "GB":
        kwargs = validated_kwargs(
            {
                "n_estimators": 100,
                "max_depth": 3,
                "random_state": 42,
            },
            "GB",
        )
        return GradientBoostingClassifier(**kwargs)
    raise ValueError(f"Unknown model type: {name!r}. Choose from: {MODEL_TYPES}")


class OscarCategoryModel:
    """Generic prediction model for any Oscar category.

    Parameters
    ----------
    data : pd.DataFrame
        Film- or nominee-level feature DataFrame.
        Must contain columns ``year_film`` and ``winner``.
    category_name : str
        Display name used in result objects (e.g. "Best Picture", "Best Actor").
    features : list[str] or str
        One of:
        - Explicit ``list[str]`` of column names.
        - ``"RFECV"`` to run Recursive Feature Elimination with CV on all
          numeric non-excluded columns (default).
        - ``"ALL"`` to use every available numeric non-excluded column.
    model_type : str
        One of ``"LR"``, ``"Ridge"``, ``"ConstrainedLR"``, ``"RF"``,
        ``"GB"``, ``"LR+RF"``.
    label_cols : list[str] or None
        Column(s) used to build display labels in backtest results.
        - ``["film"]``  →  label = row["film"]
        - ``["candidate_name", "film"]``  →  label = "Name (Film)"
        Defaults to ``["film"]``.
    exclude_cols : frozenset or None
        Additional columns to exclude from automatic feature selection.
        Merged with the built-in exclusion set.
    award_system : str
        Target award system for leakage checks (e.g. "oscars", "bafta", "sag").
    check_leakage : bool
        If True, validate feature columns against temporal leakage guardrails.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        category_name: str = "Best Picture",
        features: list[str] | str = "RFECV",
        # Default is RF for backward compat with legacy scripts.
        # Config-first pipeline (fckg/config.py) defaults to ConstrainedLR.
        model_type: str = "RF",
        label_cols: list[str] | None = None,
        exclude_cols: frozenset | None = None,
        award_system: str = "oscars",
        check_leakage: bool = False,
        estimator_params: dict[str, object] | None = None,
    ) -> None:
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}")

        required = {"year_film", "winner"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self._data = data.copy()
        self._category_name = category_name
        self._model_type = model_type
        self._label_cols = label_cols or ["film"]
        self._exclude_cols = _DEFAULT_EXCLUDE | (exclude_cols or frozenset())
        self._award_system = str(award_system).strip().lower()
        self._check_leakage = bool(check_leakage)
        self._leakage_checked = False
        self._estimator_params = dict(estimator_params or {})
        self._features = self._resolve_features(features)
        self._validate_leakage()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _available_feature_cols(self) -> list[str]:
        """All numeric columns not in the exclusion set."""
        return [
            c for c in self._data.columns
            if c not in self._exclude_cols
            and pd.api.types.is_numeric_dtype(self._data[c])
        ]

    def _resolve_features(self, spec: list[str] | str) -> list[str]:
        if spec == "ALL":
            cols = self._available_feature_cols()
            if not cols:
                raise ValueError("No numeric feature columns available")
            return cols

        if spec == "RFECV":
            return self._run_rfecv()

        # Explicit list
        features = list(spec)
        missing = [f for f in features if f not in self._data.columns]
        if missing:
            raise ValueError(f"Specified features not in DataFrame: {missing}")
        return features

    def _run_rfecv(self) -> list[str]:
        from sklearn.feature_selection import RFECV

        all_cols = self._available_feature_cols()
        if not all_cols:
            raise ValueError("No numeric feature columns available for RFECV")

        train = self._data.copy()
        if len(train) < 30:
            raise ValueError("Insufficient training data for RFECV")

        # Drop all-NaN columns before RFECV — sklearn's imputer silently skips
        # them which makes selector.support_ shorter than all_cols.
        all_cols = [c for c in all_cols if train[c].notna().any()]
        if not all_cols:
            raise ValueError("All feature columns are entirely NaN")

        # Impute + scale before RFECV so NaN columns are kept.
        prep = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X = prep.fit_transform(train[all_cols].values)
        y = train["winner"].astype(int).values
        if len(np.unique(y)) < 2:
            raise ValueError("RFECV requires both winner and non-winner rows")

        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        selector = RFECV(lr, cv=3, scoring="roc_auc", min_features_to_select=3)
        selector.fit(X, y)

        return [all_cols[i] for i in range(len(all_cols)) if selector.support_[i]]

    def _build_pipeline(self, model_type: str) -> Pipeline:
        estimator_params = self._resolve_estimator_params(model_type)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                _make_estimator(
                    model_type,
                    category_name=self._category_name,
                    features=self._features,
                    estimator_params=estimator_params,
                ),
            ),
        ])

    def _resolve_estimator_params(self, model_type: str) -> dict[str, object] | None:
        if not self._estimator_params:
            return None

        # For LR+RF allow nested per-component overrides:
        # {"LR": {...}, "RF": {...}}
        if any(isinstance(v, dict) for v in self._estimator_params.values()):
            by_key: dict[str, dict[str, object]] = {}
            for key, value in self._estimator_params.items():
                if not isinstance(value, dict):
                    continue
                by_key[str(key).strip().upper()] = dict(value)
            return by_key.get(model_type.upper())

        return dict(self._estimator_params)

    def _fit_and_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Fit imputer+scaler+model on training data, return test probabilities."""
        if self._model_type == "LR+RF":
            lr_pipe = self._build_pipeline("LR")
            rf_pipe = self._build_pipeline("RF")
            lr_pipe.fit(X_train, y_train)
            rf_pipe.fit(X_train, y_train)
            return (
                lr_pipe.predict_proba(X_test)[:, 1]
                + rf_pipe.predict_proba(X_test)[:, 1]
            ) / 2

        if self._model_type == "GB":
            n_winners = y_train.sum()
            n_total = len(y_train)
            weight_ratio = (n_total - n_winners) / max(n_winners, 1)
            sample_weight = np.where(y_train == 1, weight_ratio, 1.0)
            pipe = self._build_pipeline("GB")
            pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
            return pipe.predict_proba(X_test)[:, 1]

        pipe = self._build_pipeline(self._model_type)
        pipe.fit(X_train, y_train)
        return pipe.predict_proba(X_test)[:, 1]

    def _make_label(self, row: pd.Series) -> str:
        """Build a display label from label_cols."""
        vals = [str(row[c]) for c in self._label_cols if c in row.index]
        if not vals:
            return "(unknown)"
        if len(vals) == 1:
            return vals[0]
        return f"{vals[0]} ({', '.join(vals[1:])})"

    def _validate_leakage(self) -> None:
        """Validate selected feature columns against leakage rules once."""
        if not self._check_leakage or self._leakage_checked:
            return

        violations, warnings = check_leakage(self._features, self._award_system)
        if warnings:
            print("Leakage guardrails: warnings", file=sys.stderr)
            for warning in warnings:
                print(f"  - {warning}", file=sys.stderr)

        if violations:
            print(
                format_leakage_violations(
                    violations=violations,
                    target_system=self._award_system,
                    category_name=self._category_name,
                ),
                file=sys.stderr,
            )
            raise LeakageValidationError(
                f"Leakage validation failed with {len(violations)} violation(s)"
            )

        self._leakage_checked = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backtest(
        self,
        start_year: int = 2000,
        end_year: int | None = None,
    ) -> BacktestResult:
        """Temporal backtest: train on years < Y, evaluate on year Y.

        Parameters
        ----------
        start_year : int
            First year to evaluate (default 2000).
        end_year : int or None
            Last year to evaluate (default: latest in data).
        """
        self._validate_leakage()

        if end_year is None:
            end_year = int(self._data["year_film"].max())

        test_years = sorted(
            y for y in self._data["year_film"].unique()
            if start_year <= y <= end_year
        )

        year_results: list[YearResult] = []
        all_probs: list[dict] = []

        for yr in test_years:
            train_df = self._data[self._data["year_film"] < yr]
            test_df = self._data[self._data["year_film"] == yr].copy()

            if len(train_df) < 30 or len(test_df) == 0:
                continue
            if test_df["winner"].sum() == 0:
                continue

            proba = self._fit_and_predict(
                train_df[self._features].values,
                train_df["winner"].astype(int).values,
                test_df[self._features].values,
            )
            test_df["prob"] = proba
            test_df["y_true"] = test_df["winner"].astype(int)

            ranked = test_df.sort_values("prob", ascending=False).reset_index(drop=True)
            top_pick = self._make_label(ranked.iloc[0])
            winner_rows = ranked[ranked["winner"] == True]
            actual_winner = self._make_label(winner_rows.iloc[0])
            winner_rank = int(winner_rows.index[0]) + 1
            winner_prob = float(winner_rows.iloc[0]["prob"])

            year_results.append(YearResult(
                year=yr,
                top_pick=top_pick,
                actual_winner=actual_winner,
                winner_rank=winner_rank,
                winner_prob=winner_prob,
                correct=(winner_rank == 1),
            ))

            for _, r in test_df.iterrows():
                all_probs.append({"y_true": r["y_true"], "prob": r["prob"], "year": yr})

        probs_df = (
            pd.DataFrame(all_probs) if all_probs
            else pd.DataFrame(columns=["y_true", "prob", "year"])
        )
        metrics = compute_backtest_metrics(year_results, probs_df)

        return BacktestResult(
            config_name=self._category_name,
            model_name=self._model_type,
            features=list(self._features),
            year_results=year_results,
            all_probs=probs_df,
            **metrics,
        )

    def predict(self, target_year: int) -> PredictionResult:
        """Train on all years before target_year, predict for target_year.

        Raises ValueError if no data exists for the target year.
        """
        self._validate_leakage()

        test_df = self._data[self._data["year_film"] == target_year].copy()
        if test_df.empty:
            raise ValueError(f"No data for year {target_year}")

        train_df = self._data[self._data["year_film"] < target_year]
        if train_df.empty:
            raise ValueError(f"No training data before year {target_year}")

        proba = self._fit_and_predict(
            train_df[self._features].values,
            train_df["winner"].astype(int).values,
            test_df[self._features].values,
        )
        test_df["prob"] = proba
        predictions = test_df.sort_values("prob", ascending=False).reset_index(drop=True)

        return PredictionResult(
            year=target_year,
            config_name=self._category_name,
            model_name=self._model_type,
            features=list(self._features),
            predictions=predictions,
        )

    def feature_importance(self, target_year: int | None = None) -> pd.DataFrame:
        """Train and return feature importance scores sorted by importance.

        Parameters
        ----------
        target_year : int or None
            If given, trains only on years < target_year (matches predict() split).
            If None, trains on the full dataset.

        Returns
        -------
        pd.DataFrame with columns: ``feature``, ``importance``, ``rank``.
        """
        if target_year is not None:
            train_df = self._data[self._data["year_film"] < target_year]
        else:
            train_df = self._data

        if train_df.empty:
            return pd.DataFrame(columns=["feature", "importance", "rank"])

        X = train_df[self._features].values
        y = train_df["winner"].astype(int).values

        if self._model_type == "LR+RF":
            lr_pipe = self._build_pipeline("LR")
            rf_pipe = self._build_pipeline("RF")
            lr_pipe.fit(X, y)
            rf_pipe.fit(X, y)
            lr_imp = np.abs(lr_pipe["model"].coef_[0])
            if lr_imp.sum() > 0:
                lr_imp = lr_imp / lr_imp.sum()
            rf_imp = rf_pipe["model"].feature_importances_
            importances = (lr_imp + rf_imp) / 2

        elif self._model_type in ("LR", "Ridge", "ConstrainedLR"):
            pipe = self._build_pipeline(self._model_type)
            pipe.fit(X, y)
            importances = np.abs(pipe["model"].coef_[0])
            if importances.sum() > 0:
                importances = importances / importances.sum()

        elif self._model_type == "GB":
            n_winners = y.sum()
            n_total = len(y)
            weight_ratio = (n_total - n_winners) / max(n_winners, 1)
            sample_weight = np.where(y == 1, weight_ratio, 1.0)
            pipe = self._build_pipeline("GB")
            pipe.fit(X, y, model__sample_weight=sample_weight)
            importances = pipe["model"].feature_importances_

        else:  # RF
            pipe = self._build_pipeline("RF")
            pipe.fit(X, y)
            importances = pipe["model"].feature_importances_

        df = pd.DataFrame({"feature": self._features, "importance": importances})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df[["rank", "feature", "importance"]]

    @property
    def features(self) -> list[str]:
        """Resolved feature column list."""
        return list(self._features)

    @property
    def category_name(self) -> str:
        return self._category_name

    @property
    def model_type(self) -> str:
        return self._model_type


def grid_search(
    data: pd.DataFrame,
    feature_configs: dict[str, list[str] | str],
    category_name: str = "Best Picture",
    model_types: list[str] | None = None,
    label_cols: list[str] | None = None,
    exclude_cols: frozenset | None = None,
    start_year: int = 2000,
) -> pd.DataFrame:
    """Grid search across feature configs × model types.

    Parameters
    ----------
    data : pd.DataFrame
        Feature DataFrame passed to OscarCategoryModel.
    feature_configs : dict
        Mapping of config name → feature list or sentinel (e.g. "RFECV").
        Pass any of the existing FEATURE_CONFIGS dicts.
    category_name : str
        Forwarded to OscarCategoryModel.
    model_types : list[str] or None
        Subset of MODEL_TYPES to evaluate (default: all).
    label_cols : list[str] or None
        Forwarded to OscarCategoryModel.
    exclude_cols : frozenset or None
        Forwarded to OscarCategoryModel.
    start_year : int
        First backtest year (default 2000).

    Returns
    -------
    pd.DataFrame sorted by AUC descending, columns:
    config, model, auc, top1, top1_pct, top3, top3_pct, avg_rank, n_years, features.
    """
    if model_types is None:
        model_types = list(MODEL_TYPES)

    # Resolve RFECV once to avoid redundant computation across model types.
    rfecv_features: list[str] | None = None
    for spec in feature_configs.values():
        if spec == "RFECV":
            try:
                m = OscarCategoryModel(
                    data, category_name=category_name, features="RFECV",
                    model_type="LR", label_cols=label_cols, exclude_cols=exclude_cols,
                )
                rfecv_features = m.features
            except ValueError:
                pass
            break

    rows: list[dict] = []

    for cfg_name, spec in feature_configs.items():
        for mt in model_types:
            try:
                feats: list[str] | str
                if spec == "RFECV":
                    feats = rfecv_features if rfecv_features is not None else "RFECV"
                else:
                    feats = spec

                m = OscarCategoryModel(
                    data, category_name=category_name, features=feats,
                    model_type=mt, label_cols=label_cols, exclude_cols=exclude_cols,
                )
                result = m.backtest(start_year=start_year)
            except ValueError:
                continue

            rows.append({
                "config": cfg_name,
                "model": mt,
                "auc": result.auc,
                "top1": result.top1,
                "top1_pct": result.top1_pct,
                "top3": result.top3,
                "top3_pct": result.top3_pct,
                "avg_rank": result.avg_rank,
                "n_years": result.n_years,
                "features": ", ".join(result.features),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "config", "model", "auc", "top1", "top1_pct",
            "top3", "top3_pct", "avg_rank", "n_years", "features",
        ])

    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
