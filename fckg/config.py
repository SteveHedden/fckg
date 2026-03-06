"""Config parsing and validation for the config-first FCKG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml

from features.category_extractor import _SPECS_BY_SYSTEM
from model.category_model import MODEL_TYPES


SUPPORTED_FEATURE_MODES = {"RFECV", "ALL"}


class ConfigValidationError(ValueError):
    """Raised when a pipeline YAML config is invalid."""


def _normalize_text_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _supported_categories(award_system: str) -> list[str]:
    specs = _SPECS_BY_SYSTEM.get(award_system)
    if not specs:
        raise ConfigValidationError(
            f"Unsupported award_system {award_system!r}. Supported: {sorted(_SPECS_BY_SYSTEM)}"
        )
    return sorted(specs.keys())


def normalize_category_key(category: str, award_system: str = "oscars") -> str:
    """Normalize category aliases to canonical snake_case keys."""
    token = _normalize_text_token(str(category))
    if not token:
        raise ConfigValidationError("category must be a non-empty string")

    canonical_keys = _supported_categories(award_system)
    alias_map: dict[str, set[str]] = {}

    def add_alias(alias: str, canonical: str) -> None:
        alias_token = _normalize_text_token(alias)
        if not alias_token:
            return
        alias_map.setdefault(alias_token, set()).add(canonical)

    for canonical in canonical_keys:
        parts = canonical.split("_")
        spaced = " ".join(parts)
        hyphenated = "-".join(parts)

        add_alias(canonical, canonical)
        add_alias(spaced, canonical)
        add_alias(hyphenated, canonical)

        if canonical.startswith("best_"):
            short = canonical.removeprefix("best_")
            add_alias(short, canonical)
            add_alias(short.replace("_", " "), canonical)
            add_alias(short.replace("_", "-"), canonical)
        else:
            best_alias = f"best_{canonical}"
            add_alias(best_alias, canonical)
            add_alias(best_alias.replace("_", " "), canonical)
            add_alias(best_alias.replace("_", "-"), canonical)

    matches = alias_map.get(token, set())
    if not matches:
        allowed = ", ".join(canonical_keys)
        raise ConfigValidationError(
            f"Unknown category {category!r} for award_system {award_system!r}. "
            f"Use one of: {allowed}"
        )
    if len(matches) > 1:
        raise ConfigValidationError(
            f"Ambiguous category alias {category!r} for award_system {award_system!r}: {sorted(matches)}"
        )
    return next(iter(matches))


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{field_name} must be an integer; got {value!r}") from exc


def _ensure_str_list(raw: Any, field_name: str) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ConfigValidationError(f"{field_name} must be a non-empty list of strings")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ConfigValidationError(f"{field_name} entries must be non-empty strings; got {item!r}")
        values.append(item.strip())
    return values


def _parse_feature_spec(raw: Any) -> str | list[str]:
    """Parse features spec with precedence explicit list > RFECV > ALL."""
    if isinstance(raw, list):
        return _ensure_str_list(raw, "features")

    if isinstance(raw, str):
        mode = raw.strip().upper()
        if mode not in SUPPORTED_FEATURE_MODES:
            raise ConfigValidationError(
                f"features string must be one of {sorted(SUPPORTED_FEATURE_MODES)} or a list of feature names"
            )
        return mode

    if isinstance(raw, dict):
        explicit = raw.get("explicit")
        mode = raw.get("mode")
        rfecv_flag = bool(raw.get("rfecv", False))
        all_flag = bool(raw.get("all", False))

        if explicit is not None:
            explicit_features = _ensure_str_list(explicit, "features.explicit")
            if explicit_features:
                return explicit_features

        mode_value = None
        if mode is not None:
            if not isinstance(mode, str):
                raise ConfigValidationError("features.mode must be a string when provided")
            mode_value = mode.strip().upper()
            if mode_value not in SUPPORTED_FEATURE_MODES:
                raise ConfigValidationError(
                    f"features.mode must be one of {sorted(SUPPORTED_FEATURE_MODES)}"
                )

        if mode_value == "RFECV" or rfecv_flag:
            return "RFECV"
        if mode_value == "ALL" or all_flag:
            return "ALL"

        raise ConfigValidationError(
            "features mapping must define one of: explicit list, mode=RFECV/ALL, rfecv=true, all=true"
        )

    raise ConfigValidationError(
        "features must be either a list of feature names, a string mode, or a mapping spec"
    )


def _parse_award_system(raw: Any) -> str:
    system = "oscars" if raw is None else str(raw).strip().lower().replace("-", "_")
    if system not in _SPECS_BY_SYSTEM:
        raise ConfigValidationError(
            f"Unsupported award_system {raw!r}. Supported: {sorted(_SPECS_BY_SYSTEM)}"
        )
    return system


def _parse_hyperparameters(raw: Any, model_type: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigValidationError("model.hyperparameters must be a mapping when provided")

    if model_type != "LR+RF":
        if any(isinstance(v, dict) for v in raw.values()):
            raise ConfigValidationError(
                f"model.hyperparameters for {model_type} must be a flat mapping of estimator kwargs"
            )
        return dict(raw)

    # For LR+RF, allow either flat kwargs (applies to both) or nested LR/RF maps.
    if all(isinstance(v, dict) for v in raw.values()) and raw:
        normalized: dict[str, Any] = {}
        for key, value in raw.items():
            key_upper = str(key).strip().upper()
            if key_upper not in {"LR", "RF"}:
                raise ConfigValidationError(
                    "LR+RF nested hyperparameters may only contain LR and/or RF keys"
                )
            normalized[key_upper] = dict(value)
        return normalized
    return dict(raw)


@dataclass
class ModelConfig:
    type: str = "ConstrainedLR"
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    start_year: int
    end_year: int


@dataclass
class PipelineConfig:
    category: str
    ceremony_year: int
    model: ModelConfig
    features: str | list[str]
    label_cols: list[str]
    backtest: BacktestConfig | None
    output_dir: str
    award_system: str = "oscars"
    shap: bool = False

    @property
    def feature_mode(self) -> str:
        if isinstance(self.features, list):
            return "explicit"
        return self.features

    def to_resolved_dict(
        self,
        *,
        selected_features: list[str] | None = None,
        check_leakage: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": self.category,
            "award_system": self.award_system,
            "ceremony_year": self.ceremony_year,
            "model": {
                "type": self.model.type,
                "hyperparameters": self.model.hyperparameters,
            },
            "features": self.features,
            "feature_mode": self.feature_mode,
            "label_cols": self.label_cols,
            "backtest": None,
            "output_dir": self.output_dir,
            "check_leakage": bool(check_leakage),
            "shap": {"enabled": self.shap},
        }
        if self.backtest is not None:
            payload["backtest"] = {
                "start_year": self.backtest.start_year,
                "end_year": self.backtest.end_year,
            }
        if selected_features is not None:
            payload["selected_features"] = list(selected_features)
        return payload

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigValidationError(f"Config file not found: {config_path}")
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigValidationError(f"Invalid YAML in {config_path}: {exc}") from exc
        return cls.from_dict(loaded, source_path=config_path)

    @classmethod
    def from_dict(cls, raw: Any, source_path: str | Path | None = None) -> PipelineConfig:
        location = f" ({source_path})" if source_path else ""
        if not isinstance(raw, dict):
            raise ConfigValidationError(f"Pipeline config root must be a mapping{location}")

        award_system = _parse_award_system(raw.get("award_system"))

        if "category" not in raw:
            raise ConfigValidationError(f"Missing required field: category{location}")
        category = normalize_category_key(str(raw["category"]), award_system=award_system)

        if "ceremony_year" not in raw:
            raise ConfigValidationError(f"Missing required field: ceremony_year{location}")
        ceremony_year = _coerce_int(raw["ceremony_year"], "ceremony_year")

        model_raw = raw.get("model", {})
        if isinstance(model_raw, str):
            model_type = model_raw.strip()
            model_hparams: dict[str, Any] = {}
        elif isinstance(model_raw, dict):
            model_type = str(model_raw.get("type", "ConstrainedLR")).strip()
            model_hparams = _parse_hyperparameters(model_raw.get("hyperparameters"), model_type)
        else:
            raise ConfigValidationError("model must be a string or mapping")

        if model_type not in MODEL_TYPES:
            raise ConfigValidationError(
                f"Unsupported model type {model_type!r}. Supported: {MODEL_TYPES}"
            )
        model = ModelConfig(type=model_type, hyperparameters=model_hparams)

        if "features" not in raw:
            raise ConfigValidationError(f"Missing required field: features{location}")
        features = _parse_feature_spec(raw["features"])

        if "label_cols" not in raw:
            raise ConfigValidationError(f"Missing required field: label_cols{location}")
        label_cols = _ensure_str_list(raw["label_cols"], "label_cols")

        if "output_dir" not in raw or not str(raw["output_dir"]).strip():
            raise ConfigValidationError(f"Missing required field: output_dir{location}")
        output_dir = str(raw["output_dir"]).strip()

        backtest_raw = raw.get("backtest")
        backtest: BacktestConfig | None = None
        if backtest_raw is not None:
            if not isinstance(backtest_raw, dict):
                raise ConfigValidationError("backtest must be a mapping when provided")
            has_start = "start_year" in backtest_raw
            has_end = "end_year" in backtest_raw
            if has_start != has_end:
                raise ConfigValidationError(
                    "backtest must define both start_year and end_year, or be omitted"
                )
            if has_start and has_end:
                start = _coerce_int(backtest_raw["start_year"], "backtest.start_year")
                end = _coerce_int(backtest_raw["end_year"], "backtest.end_year")
                backtest = BacktestConfig(start_year=start, end_year=end)

        shap_raw = raw.get("shap", False)
        if isinstance(shap_raw, bool):
            shap_enabled = shap_raw
        elif isinstance(shap_raw, dict):
            shap_enabled = bool(shap_raw.get("enabled", False))
        else:
            raise ConfigValidationError("shap must be a boolean or mapping with 'enabled'")

        cfg = cls(
            category=category,
            ceremony_year=ceremony_year,
            model=model,
            features=features,
            label_cols=label_cols,
            backtest=backtest,
            output_dir=output_dir,
            award_system=award_system,
            shap=shap_enabled,
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if self.backtest is not None:
            if self.backtest.start_year > self.backtest.end_year:
                raise ConfigValidationError("backtest.start_year must be <= backtest.end_year")
            if self.backtest.end_year >= self.ceremony_year:
                raise ConfigValidationError(
                    "Invalid backtest range: must satisfy start_year <= end_year < ceremony_year"
                )
