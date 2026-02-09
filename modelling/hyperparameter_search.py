"""
Optuna-based hyperparameter optimisation for NRL prediction models.

Provides:

* ``get_search_space(model_name)`` -- returns a callable that defines
  Optuna parameter suggestions for a given model type.
* ``run_optuna_search(...)`` -- executes a full Optuna TPE search with
  MedianPruner, cross-validated scoring, and structured result output.

Search spaces are tailored for NRL-scale data (~2 600 matches, 30--80
features, binary classification).

Typical usage
-------------
>>> from modelling.hyperparameter_search import run_optuna_search
>>> from sklearn.model_selection import TimeSeriesSplit
>>> cv = TimeSeriesSplit(n_splits=5)
>>> best_params, study = run_optuna_search(
...     model_name="xgboost",
...     X=X_train,
...     y=y_train,
...     cv_splitter=cv,
...     n_trials=200,
... )
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import BaseCrossValidator

from config.settings import RANDOM_SEED

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna import Trial
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError as _exc:
    raise ImportError(
        "optuna is required for hyperparameter search. "
        "Install it with: pip install optuna"
    ) from _exc


# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------
# Each function takes an Optuna ``Trial`` and returns a ``dict`` of
# hyperparameters that can be passed to the corresponding model builder.

def _space_logistic_regression(trial: Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "penalty": "elasticnet",
        "solver": "saga",  # supports elasticnet
        "max_iter": 2000,
        "random_state": RANDOM_SEED,
    }


def _space_random_forest(trial: Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 0.7, 0.9]
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced", None]
        ),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }


def _space_xgboost(trial: Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }


def _space_lightgbm(trial: Trial) -> dict[str, Any]:
    max_depth = trial.suggest_int("max_depth", 3, 8)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int(
            "num_leaves", 15, min(2**max_depth - 1, 127)
        ),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }


def _space_catboost(trial: Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights", ["Balanced", "SqrtBalanced", None]
        ),
        "eval_metric": "Logloss",
        "random_seed": RANDOM_SEED,
        "verbose": 0,
        "allow_writing_files": False,
    }


def _space_mlp(trial: Trial) -> dict[str, Any]:
    """Search space for the ``NeuralTrainer`` + ``MLPClassifier``."""
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dims = []
    for i in range(n_layers):
        dim = trial.suggest_int(f"hidden_dim_{i}", 32, 256, step=32)
        hidden_dims.append(dim)

    return {
        "hidden_dims": hidden_dims,
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "use_batch_norm": trial.suggest_categorical("use_batch_norm", [True, False]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": 100,
        "patience": 10,
    }


def _space_lstm(trial: Trial) -> dict[str, Any]:
    """Search space for the ``NeuralTrainer`` + ``LSTMModel``."""
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 128, step=32),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        "seq_len": trial.suggest_int("seq_len", 3, 8),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": 80,
        "patience": 10,
    }


# Mapping from model name to search-space function.
_SEARCH_SPACES: dict[str, Callable[[Trial], dict[str, Any]]] = {
    "logistic_regression": _space_logistic_regression,
    "random_forest": _space_random_forest,
    "xgboost": _space_xgboost,
    "lightgbm": _space_lightgbm,
    "catboost": _space_catboost,
    "mlp": _space_mlp,
    "lstm": _space_lstm,
}

# Convenience aliases.
_SPACE_ALIASES: dict[str, str] = {
    "logreg": "logistic_regression",
    "lr": "logistic_regression",
    "rf": "random_forest",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "lgb": "lightgbm",
    "cb": "catboost",
    "cat": "catboost",
}


def get_search_space(
    model_name: str,
) -> Callable[[Trial], dict[str, Any]]:
    """Return the Optuna search-space function for *model_name*.

    Parameters
    ----------
    model_name : str
        Model identifier (case-insensitive).  Supports the same names
        and aliases as ``classical_models.get_model``, plus ``"mlp"``
        and ``"lstm"``.

    Returns
    -------
    callable
        A function ``(trial: optuna.Trial) -> dict`` that suggests
        hyperparameters.

    Raises
    ------
    ValueError
        If *model_name* is not recognised.
    """
    key = model_name.strip().lower()
    key = _SPACE_ALIASES.get(key, key)

    if key not in _SEARCH_SPACES:
        available = sorted(
            set(list(_SEARCH_SPACES.keys()) + list(_SPACE_ALIASES.keys()))
        )
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Available search spaces: {available}"
        )
    return _SEARCH_SPACES[key]


# ---------------------------------------------------------------------------
# Model builder helper
# ---------------------------------------------------------------------------

def _build_model_from_params(
    model_name: str,
    params: dict[str, Any],
    input_dim: int | None = None,
) -> Any:
    """Instantiate a model from a name and parameter dict.

    For classical models, delegates to ``classical_models.get_model``.
    For neural models, constructs the architecture and wraps it in
    ``NeuralTrainer``.
    """
    key = model_name.strip().lower()
    key = _SPACE_ALIASES.get(key, key)

    if key in ("mlp", "lstm"):
        from modelling.neural_models import (
            MLPClassifier,
            LSTMModel,
            NeuralTrainer,
        )

        trainer_keys = {"lr", "epochs", "batch_size", "patience", "weight_decay", "seq_len"}
        trainer_params = {k: v for k, v in params.items() if k in trainer_keys}
        model_params = {k: v for k, v in params.items() if k not in trainer_keys}

        if key == "mlp":
            if input_dim is None:
                raise ValueError("input_dim is required for MLP models.")
            net = MLPClassifier(input_dim=input_dim, **model_params)
        else:
            if input_dim is None:
                raise ValueError("input_dim is required for LSTM models.")
            net = LSTMModel(input_dim=input_dim, **model_params)

        return NeuralTrainer(model=net, **trainer_params)

    from modelling.classical_models import get_model
    return get_model(key, params)


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def _make_objective(
    model_name: str,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    cv_splitter: BaseCrossValidator,
    metric: str,
    input_dim: int | None,
) -> Callable[[Trial], float]:
    """Return an Optuna objective function for the given model and data."""
    space_fn = get_search_space(model_name)

    _METRIC_FNS = {
        "log_loss": lambda yt, yp: log_loss(yt, yp),
        "brier_score": lambda yt, yp: brier_score_loss(yt, yp),
    }

    if metric not in _METRIC_FNS:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose from: {sorted(_METRIC_FNS)}"
        )
    metric_fn = _METRIC_FNS[metric]

    def objective(trial: Trial) -> float:
        params = space_fn(trial)
        scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            cv_splitter.split(X, y)
        ):
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            try:
                model = _build_model_from_params(model_name, params, input_dim)
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_val)

                # Handle the case where predict_proba returns (n, 2).
                if proba.ndim == 2 and proba.shape[1] == 2:
                    p_home = proba[:, 1]
                else:
                    p_home = proba

                score = metric_fn(y_val, p_home)
                scores.append(score)

            except Exception as exc:
                logger.warning(
                    "Trial %d, fold %d failed: %s",
                    trial.number,
                    fold_idx,
                    exc,
                )
                raise optuna.exceptions.TrialPruned() from exc

            # Report intermediate result for MedianPruner.
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(scores))

    return objective


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

def run_optuna_search(
    model_name: str,
    X: pd.DataFrame | np.ndarray,
    y: ArrayLike,
    cv_splitter: BaseCrossValidator,
    n_trials: int = 200,
    metric: str = "log_loss",
    input_dim: int | None = None,
    study_name: str | None = None,
    timeout: int | None = None,
    n_jobs: int = 1,
    show_progress_bar: bool = True,
) -> tuple[dict[str, Any], "optuna.Study"]:
    """Run Optuna TPE hyperparameter search.

    Parameters
    ----------
    model_name : str
        Model identifier (same names as ``get_model`` plus ``"mlp"``,
        ``"lstm"``).
    X : pd.DataFrame or np.ndarray
        Full training features.
    y : array-like
        Binary target.
    cv_splitter : sklearn cross-validator
        Splitter defining the cross-validation folds (e.g.
        ``TimeSeriesSplit``).
    n_trials : int
        Number of Optuna trials to run.
    metric : {"log_loss", "brier_score"}
        Metric to minimise.
    input_dim : int, optional
        Number of input features (required for neural models).
    study_name : str, optional
        Name for the Optuna study (for logging / database storage).
    timeout : int, optional
        Time budget in seconds.  ``None`` for unlimited.
    n_jobs : int
        Number of parallel Optuna jobs.
    show_progress_bar : bool
        Display a tqdm progress bar during the search.

    Returns
    -------
    best_params : dict
        Best hyperparameters found.
    study : optuna.Study
        The completed Optuna study object (contains full trial history).

    Examples
    --------
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> best, study = run_optuna_search(
    ...     "xgboost", X_train, y_train,
    ...     cv_splitter=TimeSeriesSplit(n_splits=5),
    ...     n_trials=100,
    ... )
    >>> print(f"Best log_loss: {study.best_value:.4f}")
    >>> print(f"Best params: {best}")
    """
    y_np = np.asarray(y, dtype=int)

    # Infer input_dim if not provided and it's a neural model.
    key = _SPACE_ALIASES.get(model_name.strip().lower(), model_name.strip().lower())
    if key in ("mlp", "lstm") and input_dim is None:
        if isinstance(X, pd.DataFrame):
            input_dim = X.select_dtypes(include=[np.number]).shape[1]
        else:
            input_dim = np.asarray(X).shape[1]
        logger.info("Inferred input_dim=%d for neural model.", input_dim)

    objective = _make_objective(
        model_name, X, y_np, cv_splitter, metric, input_dim
    )

    sampler = TPESampler(seed=RANDOM_SEED, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    _study_name = study_name or f"nrl_{model_name}_{metric}"
    study = optuna.create_study(
        study_name=_study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    # Suppress Optuna's default logging if it is too noisy.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(
        "Starting Optuna search: model=%s, metric=%s, n_trials=%d.",
        model_name,
        metric,
        n_trials,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    best_params = study.best_params
    logger.info(
        "Optuna search complete. Best %s=%.5f across %d trials "
        "(%d pruned).",
        metric,
        study.best_value,
        len(study.trials),
        len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    )
    logger.info("Best parameters: %s", best_params)

    return best_params, study


# ---------------------------------------------------------------------------
# Convenience: summary DataFrame
# ---------------------------------------------------------------------------

def study_to_dataframe(study: "optuna.Study") -> pd.DataFrame:
    """Convert an Optuna study to a tidy DataFrame for analysis.

    Parameters
    ----------
    study : optuna.Study

    Returns
    -------
    pd.DataFrame
        One row per completed (non-pruned) trial with columns for each
        hyperparameter and the objective value.
    """
    return study.trials_dataframe(attrs=("number", "value", "params", "state"))
