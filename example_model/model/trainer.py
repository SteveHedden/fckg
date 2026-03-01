"""Best Picture prediction model: training, backtesting, and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .configs import EXCLUDE_COLUMNS, FEATURE_CONFIGS, MODEL_TYPES, create_estimator
from .evaluation import (
    BacktestResult,
    PredictionResult,
    YearResult,
    compute_backtest_metrics,
)

# Constituent columns for derived precursor aggregates.
_PRECURSOR_WIN_COLS = ["pga_winner", "sag_winner", "globe_winner", "dga_winner", "bafta_winner"]
_GUILD_SWEEP_COLS = ["pga_winner", "dga_winner", "bafta_winner"]


class BestPictureModel:
    """Train and evaluate a Best Picture prediction model.

    Parameters
    ----------
    data : pd.DataFrame
        Film-level feature DataFrame (output of FeatureExtractor).
    config : str
        Feature config name from FEATURE_CONFIGS (default: "I: Precursor awards").
    model_type : str
        One of MODEL_TYPES (default: "RF").
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: str = "I: Precursor awards",
        model_type: str = "RF",
    ):
        if config not in FEATURE_CONFIGS:
            raise ValueError(f"Unknown config: {config!r}. Choose from: {list(FEATURE_CONFIGS)}")
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model_type: {model_type!r}. Choose from: {MODEL_TYPES}")

        required = {"film", "year_film", "winner"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self._data = data.copy()
        self._config_name = config
        self._model_type = model_type

        # Compute derived columns if constituents are present
        self._add_derived_columns()

        # Resolve the feature list
        self._features = self._resolve_features()

    def _add_derived_columns(self) -> None:
        """Add num_precursor_wins and guild_sweep if constituent columns exist."""
        present = [c for c in _PRECURSOR_WIN_COLS if c in self._data.columns]
        if present:
            self._data["num_precursor_wins"] = self._data[present].sum(axis=1)
        present_guild = [c for c in _GUILD_SWEEP_COLS if c in self._data.columns]
        if len(present_guild) == len(_GUILD_SWEEP_COLS):
            self._data["guild_sweep"] = (
                (self._data["pga_winner"] == 1)
                & (self._data["dga_winner"] == 1)
                & (self._data["bafta_winner"] == 1)
            ).astype(int)

    def _resolve_features(self) -> list[str]:
        """Look up the feature list from FEATURE_CONFIGS, running RFECV if needed."""
        spec = FEATURE_CONFIGS[self._config_name]

        if spec == "RFECV":
            return self._run_rfecv()

        features = list(spec)
        missing = [f for f in features if f not in self._data.columns]
        if missing:
            raise ValueError(
                f"Config {self._config_name!r} requires columns not in data: {missing}"
            )
        return features

    def _run_rfecv(self) -> list[str]:
        """Run Recursive Feature Elimination with CV on all available columns."""
        from sklearn.feature_selection import RFECV
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        all_available = [
            c for c in self._data.columns if c not in EXCLUDE_COLUMNS
        ]
        if not all_available:
            raise ValueError("No feature columns available for RFECV")

        latest_year = self._data["year_film"].max()
        train = self._data[self._data["year_film"] < latest_year].copy()

        # Drop columns with any NaN in training set for RFECV stability
        all_available = [c for c in all_available if train[c].notna().all()]
        if not all_available:
            raise ValueError("No feature columns without NaN available for RFECV")

        scaler = StandardScaler()
        X = scaler.fit_transform(train[all_available].values)
        y = train["winner"].astype(int).values

        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        selector = RFECV(lr, cv=3, scoring="roc_auc", min_features_to_select=3)
        selector.fit(X, y)

        return [all_available[i] for i in range(len(all_available)) if selector.support_[i]]

    def _fit_and_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Fit model on training data and return test probabilities."""
        if self._model_type == "LR+RF":
            lr = create_estimator("LR")
            rf = create_estimator("RF")
            lr.fit(X_train, y_train)
            rf.fit(X_train, y_train)
            return (lr.predict_proba(X_test)[:, 1] + rf.predict_proba(X_test)[:, 1]) / 2

        if self._model_type == "GB":
            gb = create_estimator("GB")
            n_winners = y_train.sum()
            n_total = len(y_train)
            weight_ratio = (n_total - n_winners) / max(n_winners, 1)
            sample_weight = np.where(y_train == 1, weight_ratio, 1.0)
            gb.fit(X_train, y_train, sample_weight=sample_weight)
            return gb.predict_proba(X_test)[:, 1]

        model = create_estimator(self._model_type)
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]

    def backtest(
        self,
        start_year: int = 2000,
        end_year: int | None = None,
    ) -> BacktestResult:
        """Run temporal backtest: train on years < Y, test on year Y.

        Parameters
        ----------
        start_year : int
            First year to test (default: 2000).
        end_year : int or None
            Last year to test (default: max year in data).
        """
        from sklearn.preprocessing import StandardScaler

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

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[self._features].values)
            X_test = scaler.transform(test_df[self._features].values)
            y_train = train_df["winner"].astype(int).values

            proba = self._fit_and_predict(X_train, y_train, X_test)
            test_df["prob"] = proba
            test_df["y_true"] = test_df["winner"].astype(int)

            ranked = test_df.sort_values("prob", ascending=False).reset_index(drop=True)
            top_pick = ranked.iloc[0]["film"]
            winner_row = ranked[ranked["winner"] == True]
            actual_winner = winner_row.iloc[0]["film"]
            winner_rank = int(winner_row.index[0]) + 1
            winner_prob = float(winner_row.iloc[0]["prob"])

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

        probs_df = pd.DataFrame(all_probs) if all_probs else pd.DataFrame(columns=["y_true", "prob", "year"])
        metrics = compute_backtest_metrics(year_results, probs_df)

        return BacktestResult(
            config_name=self._config_name,
            model_name=self._model_type,
            features=list(self._features),
            year_results=year_results,
            all_probs=probs_df,
            **metrics,
        )

    def predict(self, target_year: int) -> PredictionResult:
        """Train on all years before target_year and predict for target_year.

        Raises ValueError if target_year has no data.
        """
        from sklearn.preprocessing import StandardScaler

        test_df = self._data[self._data["year_film"] == target_year].copy()
        if test_df.empty:
            raise ValueError(f"No data for year {target_year}")

        train_df = self._data[self._data["year_film"] < target_year]
        if train_df.empty:
            raise ValueError(f"No training data before year {target_year}")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[self._features].values)
        X_test = scaler.transform(test_df[self._features].values)
        y_train = train_df["winner"].astype(int).values

        proba = self._fit_and_predict(X_train, y_train, X_test)
        test_df["prob"] = proba
        predictions = test_df.sort_values("prob", ascending=False).reset_index(drop=True)

        return PredictionResult(
            year=target_year,
            config_name=self._config_name,
            model_name=self._model_type,
            features=list(self._features),
            predictions=predictions,
        )

    @property
    def features(self) -> list[str]:
        """The resolved feature column list."""
        return list(self._features)

    @property
    def config_name(self) -> str:
        return self._config_name

    @property
    def model_type(self) -> str:
        return self._model_type


def grid_search(
    data: pd.DataFrame,
    configs: list[str] | None = None,
    model_types: list[str] | None = None,
    start_year: int = 2000,
) -> pd.DataFrame:
    """Run backtest across all config x model_type combinations.

    Returns a DataFrame sorted by AUC descending with columns:
    config, model, auc, top1, top1_pct, top3, top3_pct, avg_rank, n_years, features.
    """
    if configs is None:
        configs = list(FEATURE_CONFIGS.keys())
    if model_types is None:
        model_types = list(MODEL_TYPES)

    # Pre-resolve RFECV once if any config uses it
    rfecv_features: list[str] | None = None
    for cfg in configs:
        if FEATURE_CONFIGS.get(cfg) == "RFECV":
            try:
                model = BestPictureModel(data, config=cfg, model_type="LR")
                rfecv_features = model.features
            except ValueError:
                pass
            break

    rows: list[dict] = []

    for cfg in configs:
        for mt in model_types:
            try:
                if FEATURE_CONFIGS.get(cfg) == "RFECV" and rfecv_features is not None:
                    # Substitute pre-resolved RFECV features via a temporary config
                    model = _make_model_with_features(data, cfg, mt, rfecv_features)
                else:
                    model = BestPictureModel(data, config=cfg, model_type=mt)
                result = model.backtest(start_year=start_year)
            except ValueError:
                continue

            rows.append({
                "config": cfg,
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

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    return df


def _make_model_with_features(
    data: pd.DataFrame,
    config_name: str,
    model_type: str,
    features: list[str],
) -> BestPictureModel:
    """Create a BestPictureModel with pre-resolved features (for RFECV caching)."""
    model = object.__new__(BestPictureModel)
    model._data = data.copy()
    model._config_name = config_name
    model._model_type = model_type
    # Add derived columns
    model._add_derived_columns()
    # Validate features exist
    missing = [f for f in features if f not in model._data.columns]
    if missing:
        raise ValueError(f"Pre-resolved features missing from data: {missing}")
    model._features = list(features)
    return model
