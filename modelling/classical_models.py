"""
Classical machine-learning models for NRL match winner prediction.

Provides factory functions that return configured (but untrained) estimators
with sensible default hyperparameters tuned for NRL-scale tabular data
(~2 600 matches, 30--80 features).  Each builder accepts an optional
``params`` dict to override any default.

The ``get_model`` factory resolves a model by name, making it easy to
iterate over model types in training and hyper-parameter search scripts.

Typical usage
-------------
>>> from modelling.classical_models import get_model
>>> model = get_model("xgboost")
>>> model.fit(X_train, y_train)
>>> probas = model.predict_proba(X_test)
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from config.settings import RANDOM_SEED

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------
# Tuned for ~2 600 rows, 30-80 features, binary classification.

_LOGISTIC_DEFAULTS: dict[str, Any] = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
}

_RANDOM_FOREST_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "min_samples_split": 20,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

_XGBOOST_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

_LIGHTGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary",
    "metric": "binary_logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

_CATBOOST_DEFAULTS: dict[str, Any] = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "border_count": 128,
    "auto_class_weights": "Balanced",
    "eval_metric": "Logloss",
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "allow_writing_files": False,
}


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_logistic_regression(
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Return a ``Pipeline(StandardScaler -> LogisticRegression)``.

    Parameters
    ----------
    params : dict, optional
        Overrides for ``LogisticRegression`` hyper-parameters.  Keys should
        match ``LogisticRegression`` constructor arguments.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An untrained pipeline ready for ``.fit(X, y)``.
    """
    cfg = {**_LOGISTIC_DEFAULTS, **(params or {})}
    logger.debug("Building LogisticRegression with params: %s", cfg)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**cfg)),
        ]
    )


def build_random_forest(
    params: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """Return a configured ``RandomForestClassifier``.

    Parameters
    ----------
    params : dict, optional
        Overrides for ``RandomForestClassifier`` hyper-parameters.

    Returns
    -------
    RandomForestClassifier
    """
    cfg = {**_RANDOM_FOREST_DEFAULTS, **(params or {})}
    logger.debug("Building RandomForestClassifier with params: %s", cfg)
    return RandomForestClassifier(**cfg)


def build_xgboost(
    params: dict[str, Any] | None = None,
) -> "XGBClassifier":
    """Return a configured ``XGBClassifier``.

    Parameters
    ----------
    params : dict, optional
        Overrides for ``XGBClassifier`` hyper-parameters.

    Returns
    -------
    xgboost.XGBClassifier
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for build_xgboost(). "
            "Install it with: pip install xgboost"
        ) from exc

    cfg = {**_XGBOOST_DEFAULTS, **(params or {})}
    logger.debug("Building XGBClassifier with params: %s", cfg)
    return XGBClassifier(**cfg)


def build_lightgbm(
    params: dict[str, Any] | None = None,
) -> "LGBMClassifier":
    """Return a configured ``LGBMClassifier``.

    Parameters
    ----------
    params : dict, optional
        Overrides for ``LGBMClassifier`` hyper-parameters.

    Returns
    -------
    lightgbm.LGBMClassifier
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for build_lightgbm(). "
            "Install it with: pip install lightgbm"
        ) from exc

    cfg = {**_LIGHTGBM_DEFAULTS, **(params or {})}
    logger.debug("Building LGBMClassifier with params: %s", cfg)
    return LGBMClassifier(**cfg)


def build_catboost(
    params: dict[str, Any] | None = None,
) -> "CatBoostClassifier":
    """Return a configured ``CatBoostClassifier``.

    Parameters
    ----------
    params : dict, optional
        Overrides for ``CatBoostClassifier`` hyper-parameters.

    Returns
    -------
    catboost.CatBoostClassifier
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "catboost is required for build_catboost(). "
            "Install it with: pip install catboost"
        ) from exc

    cfg = {**_CATBOOST_DEFAULTS, **(params or {})}
    logger.debug("Building CatBoostClassifier with params: %s", cfg)
    return CatBoostClassifier(**cfg)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_MODEL_BUILDERS: dict[str, Any] = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "catboost": build_catboost,
}

# Convenient aliases
_MODEL_ALIASES: dict[str, str] = {
    "logreg": "logistic_regression",
    "lr": "logistic_regression",
    "rf": "random_forest",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "lgb": "lightgbm",
    "cb": "catboost",
    "cat": "catboost",
}


def get_model(
    name: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Factory: return a configured (untrained) model by name.

    Parameters
    ----------
    name : str
        Model identifier.  Supported values (case-insensitive):
        ``"logistic_regression"`` (alias ``"logreg"``, ``"lr"``),
        ``"random_forest"`` (alias ``"rf"``),
        ``"xgboost"`` (alias ``"xgb"``),
        ``"lightgbm"`` (alias ``"lgbm"``, ``"lgb"``),
        ``"catboost"`` (alias ``"cb"``, ``"cat"``).
    params : dict, optional
        Hyper-parameter overrides passed to the builder function.

    Returns
    -------
    estimator
        An untrained scikit-learn-compatible estimator.

    Raises
    ------
    ValueError
        If *name* is not a recognised model name or alias.

    Examples
    --------
    >>> model = get_model("xgboost", {"max_depth": 4, "n_estimators": 300})
    >>> model = get_model("lgbm")
    """
    key = name.strip().lower()
    key = _MODEL_ALIASES.get(key, key)

    if key not in _MODEL_BUILDERS:
        available = sorted(
            set(list(_MODEL_BUILDERS.keys()) + list(_MODEL_ALIASES.keys()))
        )
        raise ValueError(
            f"Unknown model name '{name}'. "
            f"Available: {available}"
        )

    builder = _MODEL_BUILDERS[key]
    logger.info("Building model '%s' (resolved from '%s').", key, name)
    return builder(params)


def list_models() -> list[str]:
    """Return the canonical names of all available models."""
    return sorted(_MODEL_BUILDERS.keys())
