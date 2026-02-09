"""
Model interpretability tools for NRL match winner prediction.

Provides functions for understanding *why* a model makes the predictions
it does:

* **compute_shap_values** -- SHAP analysis for tree-based models.
* **plot_feature_importance** -- bar chart of feature importance.
* **plot_shap_summary** -- SHAP beeswarm summary plot.
* **plot_partial_dependence** -- partial dependence plots for selected
  features.

All plotting functions save figures to ``outputs/figures/`` by default.

Typical usage
-------------
>>> from modelling.interpretability import compute_shap_values, plot_shap_summary
>>> shap_values = compute_shap_values(my_xgb, X_test, feature_names)
>>> plot_shap_summary(shap_values, X_test, feature_names)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from config.settings import OUTPUTS_DIR

logger = logging.getLogger(__name__)

_FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. SHAP Value Computation
# ============================================================================

def compute_shap_values(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None = None,
    max_samples: int = 500,
) -> np.ndarray:
    """Compute SHAP values for a model.

    Automatically selects the appropriate SHAP explainer:

    * ``TreeExplainer`` for tree-based models (XGBoost, LightGBM,
      CatBoost, RandomForest).
    * ``KernelExplainer`` as a fallback for other model types (slower).

    Parameters
    ----------
    model : estimator
        A fitted model.  For pipelines, the final estimator is extracted
        automatically.
    X : pd.DataFrame or np.ndarray
        Feature matrix used for computing SHAP values.
    feature_names : list[str], optional
        Names for each feature column.  If *X* is a DataFrame, column
        names are used by default.
    max_samples : int
        Maximum number of background samples for ``KernelExplainer``
        (ignored by ``TreeExplainer``).

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        SHAP values for the positive class (home win).

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for SHAP analysis. "
            "Install it with: pip install shap"
        ) from exc

    # Unwrap sklearn Pipeline if necessary.
    estimator = _unwrap_pipeline(model)
    X_array = _ensure_array(X)

    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)

    # Choose explainer.
    if _is_tree_model(estimator):
        logger.info("Using TreeExplainer for %s.", type(estimator).__name__)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_array)

        # TreeExplainer may return a list (one per class), a 3-D array
        # (n_samples, n_features, n_classes) in SHAP >= 0.50, or a 2-D
        # array.  We always want the positive-class slice.
        if isinstance(shap_values, list):
            # Legacy SHAP: list of arrays, one per class.
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # SHAP >= 0.50: (n_samples, n_features, n_classes).
            shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]
    else:
        logger.info(
            "Using KernelExplainer for %s (may be slow).",
            type(estimator).__name__,
        )
        # Sub-sample the background data for efficiency.
        n_bg = min(max_samples, X_array.shape[0])
        bg_indices = np.random.default_rng(42).choice(
            X_array.shape[0], size=n_bg, replace=False
        )
        background = X_array[bg_indices]

        def predict_fn(x: np.ndarray) -> np.ndarray:
            return model.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_array, nsamples=100)

    logger.info(
        "SHAP values computed: shape=%s.",
        np.asarray(shap_values).shape,
    )
    return np.asarray(shap_values)


# ============================================================================
# 2. Feature Importance Plot
# ============================================================================

def plot_feature_importance(
    model: Any,
    feature_names: Sequence[str],
    top_n: int = 20,
    importance_type: str = "auto",
    save_path: str | Path | None = None,
    show: bool = False,
) -> "matplotlib.figure.Figure":
    """Bar chart of feature importance from a tree-based model.

    Parameters
    ----------
    model : estimator
        A fitted model with ``feature_importances_`` attribute, or a
        pipeline containing one.
    feature_names : list[str]
        Names for each feature.
    top_n : int
        Number of top features to display.
    importance_type : str
        For XGBoost models: ``"weight"``, ``"gain"``, ``"cover"``, or
        ``"auto"`` (uses ``feature_importances_``).
    save_path : str or Path, optional
        Where to save the figure.  Defaults to
        ``outputs/figures/feature_importance.png``.
    show : bool
        Whether to display the plot interactively.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    estimator = _unwrap_pipeline(model)

    # Extract importances.
    if importance_type != "auto" and hasattr(estimator, "get_booster"):
        # XGBoost-specific importance types.
        booster = estimator.get_booster()
        raw = booster.get_score(importance_type=importance_type)
        # Map feature index names ("f0", "f1", ...) to actual names.
        importances = np.zeros(len(feature_names))
        for fname, score in raw.items():
            idx = int(fname[1:]) if fname.startswith("f") else None
            if idx is not None and idx < len(feature_names):
                importances[idx] = score
    elif hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
    else:
        raise AttributeError(
            f"{type(estimator).__name__} does not have 'feature_importances_'. "
            "Use compute_shap_values for model-agnostic importance."
        )

    if len(importances) != len(feature_names):
        raise ValueError(
            f"Length mismatch: {len(importances)} importances vs "
            f"{len(feature_names)} feature names."
        )

    # Sort and take top N.
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in indices]
    sorted_imp = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_imp, color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances ({type(estimator).__name__})")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()

    if save_path is None:
        save_path = _FIGURES_DIR / "feature_importance.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Feature importance plot saved to %s", save_path)

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ============================================================================
# 3. SHAP Summary Plot
# ============================================================================

def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None = None,
    top_n: int = 20,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Generate a SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values : np.ndarray of shape (n_samples, n_features)
        SHAP values (from ``compute_shap_values``).
    X : pd.DataFrame or np.ndarray
        The feature matrix corresponding to *shap_values*.
    feature_names : list[str], optional
        Feature labels.
    top_n : int
        Number of top features to include.
    save_path : str or Path, optional
        Defaults to ``outputs/figures/shap_summary.png``.
    show : bool
        Whether to display interactively.
    """
    import matplotlib.pyplot as plt

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required. Install it with: pip install shap"
        ) from exc

    X_array = _ensure_array(X)

    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)

    if feature_names is not None:
        X_display = pd.DataFrame(X_array, columns=feature_names)
    else:
        X_display = X_array

    fig = plt.figure(figsize=(10, max(6, top_n * 0.35)))
    shap.summary_plot(
        shap_values,
        X_display,
        max_display=top_n,
        show=False,
    )

    if save_path is None:
        save_path = _FIGURES_DIR / "shap_summary.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("SHAP summary plot saved to %s", save_path)

    if show:
        plt.show()

    plt.close("all")


# ============================================================================
# 4. Partial Dependence Plots
# ============================================================================

def plot_partial_dependence(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    features: Sequence[int] | Sequence[str],
    feature_names: Sequence[str] | None = None,
    grid_resolution: int = 50,
    save_path: str | Path | None = None,
    show: bool = False,
) -> "matplotlib.figure.Figure":
    """Partial dependence plots for selected features.

    Parameters
    ----------
    model : estimator
        A fitted model with ``predict_proba``.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    features : list[int] or list[str]
        Feature indices or names to plot.  Up to 6 features recommended.
    feature_names : list[str], optional
        Feature labels.
    grid_resolution : int
        Number of points in the PDP grid per feature.
    save_path : str or Path, optional
        Defaults to ``outputs/figures/partial_dependence.png``.
    show : bool
        Whether to display interactively.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay

    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)

    # Resolve feature names to indices if necessary.
    resolved_features: list[int | str] = []
    for feat in features:
        if isinstance(feat, str) and feature_names is not None:
            if feat in feature_names:
                resolved_features.append(feature_names.index(feat))
            else:
                raise ValueError(
                    f"Feature '{feat}' not found in feature_names."
                )
        else:
            resolved_features.append(feat)

    # Unwrap pipeline for compatibility with sklearn PDP.
    estimator = _unwrap_pipeline(model)
    X_array = _ensure_array(X)

    n_features = len(resolved_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    # Use sklearn's PartialDependenceDisplay if the model is compatible.
    try:
        display = PartialDependenceDisplay.from_estimator(
            estimator,
            X_array,
            features=resolved_features,
            feature_names=feature_names,
            grid_resolution=grid_resolution,
            ax=axes.ravel()[:n_features],
            kind="average",
        )
    except Exception:
        # Fallback: compute PDP manually for models not supported by sklearn.
        logger.info(
            "sklearn PDP failed; computing partial dependence manually."
        )
        _manual_pdp(
            model, X_array, resolved_features, feature_names,
            grid_resolution, axes, n_features,
        )

    # Hide unused axes.
    for idx in range(n_features, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Partial Dependence Plots", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path is None:
        save_path = _FIGURES_DIR / "partial_dependence.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Partial dependence plot saved to %s", save_path)

    if show:
        plt.show()

    plt.close(fig)
    return fig


def _manual_pdp(
    model: Any,
    X: np.ndarray,
    features: list[int],
    feature_names: Sequence[str] | None,
    grid_resolution: int,
    axes: np.ndarray,
    n_features: int,
) -> None:
    """Compute and plot PDP manually (model-agnostic fallback)."""
    n_cols = axes.shape[1]

    for plot_idx, feat_idx in enumerate(features):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row][col]

        feat_values = np.linspace(
            X[:, feat_idx].min(),
            X[:, feat_idx].max(),
            grid_resolution,
        )
        pdp_values = np.empty(grid_resolution)

        for i, val in enumerate(feat_values):
            X_modified = X.copy()
            X_modified[:, feat_idx] = val
            proba = model.predict_proba(X_modified)[:, 1]
            pdp_values[i] = proba.mean()

        feat_name = (
            feature_names[feat_idx] if feature_names is not None
            else f"Feature {feat_idx}"
        )
        ax.plot(feat_values, pdp_values, color="#1f77b4", linewidth=2)
        ax.set_xlabel(feat_name)
        ax.set_ylabel("Mean predicted P(home win)")
        ax.set_title(f"PDP: {feat_name}")
        ax.grid(True, alpha=0.3)


# ============================================================================
# Internal helpers
# ============================================================================

def _unwrap_pipeline(model: Any) -> Any:
    """If *model* is a sklearn Pipeline, return the final estimator."""
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _ensure_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert to a plain numpy array."""
    if isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)


def _is_tree_model(estimator: Any) -> bool:
    """Check if the estimator is a tree-based model supported by TreeExplainer."""
    tree_types = (
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
        "CatBoostClassifier",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "XGBRegressor",
        "LGBMRegressor",
        "CatBoostRegressor",
    )
    return type(estimator).__name__ in tree_types
