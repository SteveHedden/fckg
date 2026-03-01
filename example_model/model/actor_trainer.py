"""Best Actor prediction model: training, backtesting, and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .actor_configs import (
    EXCLUDE_COLUMNS_ACTOR,
    FEATURE_CONFIGS_ACTOR,
    MODEL_TYPES,
    create_estimator,
)
from .evaluation import (
    BacktestResult,
    PredictionResult,
    YearResult,
    compute_backtest_metrics,
)


class BestActorModel:
    """Train and evaluate a Best Actor prediction model."""

    def __init__(
        self,
        data: pd.DataFrame,
        config: str = "A: Canonical prior history",
        model_type: str = "RF",
    ):
        if config not in FEATURE_CONFIGS_ACTOR:
            raise ValueError(
                f"Unknown config: {config!r}. Choose from: {list(FEATURE_CONFIGS_ACTOR)}"
            )
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model_type: {model_type!r}. Choose from: {MODEL_TYPES}")

        required = {"candidate_id", "candidate_name", "film", "year_film", "winner"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self._data = data.copy()
        self._config_name = config
        self._model_type = model_type
        self._features = self._resolve_features()

    def _resolve_features(self) -> list[str]:
        spec = FEATURE_CONFIGS_ACTOR[self._config_name]
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
        from sklearn.feature_selection import RFECV
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        all_available = [c for c in self._data.columns if c not in EXCLUDE_COLUMNS_ACTOR]
        if not all_available:
            raise ValueError("No feature columns available for RFECV")

        latest_year = self._data["year_film"].max()
        train = self._data[self._data["year_film"] < latest_year].copy()
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
            top = ranked.iloc[0]
            winner_row = ranked[ranked["winner"] == True].iloc[0]

            year_results.append(
                YearResult(
                    year=yr,
                    top_pick=f"{top['candidate_name']} ({top['film']})",
                    actual_winner=f"{winner_row['candidate_name']} ({winner_row['film']})",
                    winner_rank=int(ranked[ranked["winner"] == True].index[0]) + 1,
                    winner_prob=float(winner_row["prob"]),
                    correct=(bool(top["winner"])),
                )
            )

            for _, r in test_df.iterrows():
                all_probs.append({"y_true": r["y_true"], "prob": r["prob"], "year": yr})

        probs_df = (
            pd.DataFrame(all_probs)
            if all_probs
            else pd.DataFrame(columns=["y_true", "prob", "year"])
        )
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
        return list(self._features)

    @property
    def config_name(self) -> str:
        return self._config_name

    @property
    def model_type(self) -> str:
        return self._model_type


def _make_model_with_features(
    data: pd.DataFrame,
    config_name: str,
    model_type: str,
    features: list[str],
) -> BestActorModel:
    model = BestActorModel(data, config="A: Canonical prior history", model_type=model_type)
    model._config_name = config_name
    model._features = list(features)
    return model


def grid_search(
    data: pd.DataFrame,
    configs: list[str] | None = None,
    model_types: list[str] | None = None,
    start_year: int = 2000,
) -> pd.DataFrame:
    if configs is None:
        configs = list(FEATURE_CONFIGS_ACTOR.keys())
    if model_types is None:
        model_types = list(MODEL_TYPES)

    rfecv_features: list[str] | None = None
    for cfg in configs:
        if FEATURE_CONFIGS_ACTOR.get(cfg) == "RFECV":
            try:
                model = BestActorModel(data, config=cfg, model_type="LR")
                rfecv_features = model.features
            except ValueError:
                pass
            break

    rows: list[dict] = []
    for cfg in configs:
        for mt in model_types:
            try:
                if FEATURE_CONFIGS_ACTOR.get(cfg) == "RFECV" and rfecv_features is not None:
                    model = _make_model_with_features(data, cfg, mt, rfecv_features)
                else:
                    model = BestActorModel(data, config=cfg, model_type=mt)
                result = model.backtest(start_year=start_year)
            except ValueError:
                continue

            rows.append(
                {
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
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "config", "model", "auc", "top1", "top1_pct",
                "top3", "top3_pct", "avg_rank", "n_years", "features",
            ]
        )
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)

