"""Visualization helpers: feature importance bar charts and SHAP plots.

Requirements
────────────
  pip install shap matplotlib
"""

from __future__ import annotations

import os
import re

import numpy as np

# Set non-interactive backend before any matplotlib import.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_stem(label: str) -> str:
    """Convert a nominee label to a safe filename stem (max 60 chars)."""
    return re.sub(r"[^\w\-]", "_", label).strip("_")[:60]


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(
    features: list[str],
    importances: np.ndarray,
    output_path,
    directions: np.ndarray | None = None,
    title: str = "Feature Importance",
) -> None:
    """Horizontal bar chart sorted by absolute importance, coloured by direction.

    Parameters
    ----------
    features    : feature names in the same order as *importances*
    importances : raw values — signed LR coefficients or unsigned RF/GB scores
    directions  : optional array of signs (+1 / -1) for each feature.
                  If None, sign is inferred from importances themselves.
                  Recommended: pass model-derived contribution signs
                  (e.g. mean SHAP sign) so colours reflect direction.
    output_path : destination PNG path
    """
    # Determine sign for colouring
    if directions is not None:
        signs = np.sign(directions)
    else:
        signs = np.sign(importances)

    # Sort by absolute importance ascending (so most important is at top)
    order = np.argsort(np.abs(importances))
    feat_sorted = [features[i] for i in order]
    imp_sorted = np.abs(importances[order])   # always plot positive bar length
    sign_sorted = signs[order]

    colors = ["#2ecc71" if s >= 0 else "#e74c3c" for s in sign_sorted]

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.55)))
    ax.barh(feat_sorted, imp_sorted, color=colors, edgecolor="white", height=0.65)
    ax.legend(handles=[
        mpatches.Patch(color="#2ecc71", label="Positive (↑ win probability)"),
        mpatches.Patch(color="#e74c3c", label="Negative (↓ win probability)"),
    ], loc="lower right", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance (magnitude)", fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(
    pipe,
    X_train: np.ndarray,
    X_test: np.ndarray,
    model_type: str,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Compute SHAP values for test nominees.

    Transforms data through the pipeline's preprocessing steps, then applies
    an appropriate SHAP explainer to the final estimator.

    For LR+RF ensembles the LR component is used (more interpretable).

    Returns
    -------
    sv   : np.ndarray (n_test, n_features) — positive-class SHAP values
    base : float — expected value baseline
    X_t  : np.ndarray — preprocessed X_test (used as waterfall feature data)
    """
    import shap

    pre = pipe[:-1]
    X_train_t = pre.transform(X_train)
    X_test_t = pre.transform(X_test)
    mdl = pipe["model"]

    if model_type in ("LR", "Ridge", "ConstrainedLR", "LR+RF"):
        exp = shap.LinearExplainer(mdl, X_train_t)
        sv = np.array(exp.shap_values(X_test_t))
        if sv.ndim == 3:
            sv = sv[1]
        ev = np.atleast_1d(exp.expected_value)
        base = float(ev[-1])

    else:  # RF or GB — TreeExplainer
        exp = shap.TreeExplainer(mdl)
        sv_raw = exp.shap_values(X_test_t)
        sv = np.array(sv_raw[1] if isinstance(sv_raw, list) else sv_raw)
        # RF TreeExplainer can return shape (n, f, 2) — take positive class
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        ev = np.atleast_1d(exp.expected_value)
        base = float(ev[-1])

    return sv, base, X_test_t


def compute_direction_signs_from_shap(
    pipe,
    X_ref: np.ndarray,
    model_type: str,
    max_samples: int = 500,
) -> np.ndarray | None:
    """Return per-feature direction signs from mean SHAP contribution.

    Signs are computed as sign(mean SHAP value) across a reference sample.
    Positive means the feature tends to push win probability up; negative means
    it tends to push it down.
    """
    if X_ref.size == 0:
        return None

    X_eval = X_ref
    if X_ref.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_ref.shape[0], size=max_samples, replace=False)
        X_eval = X_ref[idx]

    try:
        sv, _, _ = compute_shap(pipe, X_eval, X_eval, model_type)
    except Exception:
        return None

    if sv.ndim != 2 or sv.shape[1] == 0:
        return None
    return np.sign(np.nanmean(sv, axis=0))


def plot_shap_summary(
    sv: np.ndarray,
    X_t: np.ndarray,
    features: list[str],
    output_path,
    title: str = "SHAP Summary",
) -> None:
    """Beeswarm/dot summary plot across all nominees — shows all features."""
    import shap

    # Scale height with feature count so nothing gets cropped
    height = max(4, len(features) * 0.45 + 1.5)

    shap.summary_plot(
        sv, X_t, feature_names=features,
        max_display=len(features),          # never truncate
        plot_size=(10, height),             # explicit size scales with n_features
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_shap_waterfall(
    sv_row: np.ndarray,
    base: float,
    X_row: np.ndarray,
    features: list[str],
    label: str,
    output_path,
) -> None:
    """Waterfall plot for a single nominee."""
    import shap

    explanation = shap.Explanation(
        values=sv_row,
        base_values=base,
        data=X_row,
        feature_names=features,
    )
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    fig.suptitle(label, fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
