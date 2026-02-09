"""
Ensemble methods for NRL match winner prediction.

Provides three ensembling strategies:

* **VotingEnsemble** -- hard and soft voting across base models.
* **StackingEnsemble** -- trains a meta-learner (logistic regression by
  default) on out-of-fold base-model predictions.
* **OddsBlender** -- blends a model's predicted probability with the
  odds-implied probability using weights optimised via log-loss
  minimisation on a calibration set.

All classes implement the standard ``fit``, ``predict``, and
``predict_proba`` interface for drop-in use with the backtesting harness.

Typical usage
-------------
>>> from modelling.ensemble import StackingEnsemble
>>> from modelling.classical_models import get_model
>>> base = [get_model("xgboost"), get_model("lightgbm"), get_model("logreg")]
>>> stack = StackingEnsemble(base_models=base, n_folds=5)
>>> stack.fit(X_train, y_train)
>>> proba = stack.predict_proba(X_test)
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from config.settings import RANDOM_SEED

logger = logging.getLogger(__name__)


# ============================================================================
# Helper
# ============================================================================

def _collect_probas(
    models: Sequence,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Stack ``predict_proba`` outputs from multiple models.

    Returns
    -------
    np.ndarray of shape (n_models, n_samples, 2)
    """
    return np.array([m.predict_proba(X) for m in models])


# ============================================================================
# 1. Voting Ensemble
# ============================================================================

class VotingEnsemble:
    """Hard or soft voting ensemble over base models.

    Parameters
    ----------
    base_models : list
        Untrained (or pre-trained) estimators with ``fit``,
        ``predict``, and ``predict_proba``.
    voting : {"soft", "hard"}
        ``"soft"`` averages predicted probabilities; ``"hard"`` uses
        majority vote on predicted class labels.
    weights : list[float], optional
        Per-model weights.  ``None`` gives equal weight.
    refit : bool
        If ``True`` (default), ``fit`` trains each base model.  Set to
        ``False`` if models are already fitted.
    """

    def __init__(
        self,
        base_models: Sequence,
        voting: str = "soft",
        weights: Sequence[float] | None = None,
        refit: bool = True,
    ) -> None:
        if voting not in ("soft", "hard"):
            raise ValueError(f"voting must be 'soft' or 'hard', got '{voting}'")
        self.base_models = list(base_models)
        self.voting = voting
        self.weights = (
            np.array(weights, dtype=float) if weights is not None
            else np.ones(len(self.base_models))
        )
        self.refit = refit
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: ArrayLike,
    ) -> "VotingEnsemble":
        """Fit all base models on the training data.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Binary target.

        Returns
        -------
        self
        """
        if self.refit:
            for i, model in enumerate(self.base_models):
                logger.info(
                    "VotingEnsemble: fitting base model %d/%d (%s).",
                    i + 1,
                    len(self.base_models),
                    type(model).__name__,
                )
                model.fit(X, y)
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return weighted average class probabilities (soft voting).

        For hard voting this still returns aggregated probabilities to
        enable calibration and logging, but ``predict`` uses majority vote.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        if not self.is_fitted_:
            raise RuntimeError("VotingEnsemble has not been fitted yet.")

        all_proba = _collect_probas(self.base_models, X)  # (n_models, n, 2)
        # Weighted average.
        w = self.weights / self.weights.sum()
        avg_proba = np.tensordot(w, all_proba, axes=([0], [0]))
        return avg_proba

    # ------------------------------------------------------------------
    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if self.voting == "soft":
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)

        # Hard voting: majority of individual predictions.
        preds = np.array([m.predict(X) for m in self.base_models])  # (n_models, n)
        w = self.weights / self.weights.sum()
        weighted_votes = np.tensordot(w, preds, axes=([0], [0]))
        return (weighted_votes >= 0.5).astype(int)

    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        return {
            "voting": self.voting,
            "weights": self.weights.tolist(),
            "refit": self.refit,
        }

    def set_params(self, **params: Any) -> "VotingEnsemble":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        names = [type(m).__name__ for m in self.base_models]
        return (
            f"VotingEnsemble(models={names}, voting='{self.voting}')"
        )


# ============================================================================
# 2. Stacking Ensemble
# ============================================================================

class StackingEnsemble:
    """Train a meta-learner on out-of-fold base-model predictions.

    During ``fit``, each base model generates out-of-fold predicted
    probabilities via stratified K-fold cross-validation.  These
    predictions become input features for a logistic-regression
    meta-learner.  All base models are then re-trained on the full
    training set.

    Parameters
    ----------
    base_models : list
        Untrained estimators.
    meta_learner : estimator, optional
        The stacking meta-learner.  Defaults to ``LogisticRegression``.
    n_folds : int
        Number of CV folds for generating out-of-fold predictions.
    use_proba : bool
        If ``True``, feed predicted probabilities to the meta-learner;
        if ``False``, feed predicted class labels.
    passthrough : bool
        If ``True``, concatenate original features with meta-features.
    """

    def __init__(
        self,
        base_models: Sequence,
        meta_learner: Any | None = None,
        n_folds: int = 5,
        use_proba: bool = True,
        passthrough: bool = False,
    ) -> None:
        self.base_models = list(base_models)
        self.meta_learner = meta_learner or LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_SEED
        )
        self.n_folds = n_folds
        self.use_proba = use_proba
        self.passthrough = passthrough
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    def _generate_oof_predictions(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
    ) -> np.ndarray:
        """Generate out-of-fold predictions for every base model.

        Returns
        -------
        np.ndarray of shape (n_samples, n_meta_features)
            If ``use_proba`` is True, each model contributes the home-win
            probability (1 column).  Otherwise each contributes a binary
            prediction.
        """
        import copy as _copy

        n_samples = len(y)
        n_models = len(self.base_models)
        oof = np.zeros((n_samples, n_models), dtype=float)

        kf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=RANDOM_SEED,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            if isinstance(X, pd.DataFrame):
                X_fold_train = X.iloc[train_idx]
                X_fold_val = X.iloc[val_idx]
            else:
                X_fold_train = X[train_idx]
                X_fold_val = X[val_idx]
            y_fold_train = y[train_idx]

            for m_idx, model in enumerate(self.base_models):
                clone = _copy.deepcopy(model)
                clone.fit(X_fold_train, y_fold_train)

                if self.use_proba:
                    preds = clone.predict_proba(X_fold_val)[:, 1]
                else:
                    preds = clone.predict(X_fold_val).astype(float)

                oof[val_idx, m_idx] = preds

            logger.debug(
                "StackingEnsemble: completed fold %d/%d.",
                fold_idx + 1,
                self.n_folds,
            )

        return oof

    # ------------------------------------------------------------------
    def _build_meta_features(
        self,
        X: np.ndarray | pd.DataFrame,
        oof: np.ndarray,
    ) -> np.ndarray:
        """Optionally concatenate original features with OOF predictions."""
        if self.passthrough:
            X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            return np.hstack([oof, X_np.astype(float)])
        return oof

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: ArrayLike,
    ) -> "StackingEnsemble":
        """Fit the stacking ensemble.

        1. Generate out-of-fold predictions from each base model.
        2. Fit the meta-learner on those predictions.
        3. Re-fit every base model on the full training set.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Binary target.

        Returns
        -------
        self
        """
        y_np = np.asarray(y, dtype=int)

        logger.info(
            "StackingEnsemble: generating OOF predictions "
            "(%d folds, %d base models).",
            self.n_folds,
            len(self.base_models),
        )
        oof = self._generate_oof_predictions(X, y_np)
        meta_X = self._build_meta_features(X, oof)

        logger.info("StackingEnsemble: fitting meta-learner.")
        self.meta_learner.fit(meta_X, y_np)

        # Re-train base models on the full training set.
        logger.info(
            "StackingEnsemble: re-fitting %d base models on full data.",
            len(self.base_models),
        )
        for model in self.base_models:
            model.fit(X, y_np)

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def _meta_features_from_base(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Generate meta-features from fitted base models."""
        cols = []
        for model in self.base_models:
            if self.use_proba:
                cols.append(model.predict_proba(X)[:, 1])
            else:
                cols.append(model.predict(X).astype(float))
        oof = np.column_stack(cols)
        return self._build_meta_features(X, oof)

    # ------------------------------------------------------------------
    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return meta-learner class probabilities.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        if not self.is_fitted_:
            raise RuntimeError("StackingEnsemble has not been fitted yet.")

        meta_X = self._meta_features_from_base(X)
        return self.meta_learner.predict_proba(meta_X)

    # ------------------------------------------------------------------
    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return predicted class labels.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_folds": self.n_folds,
            "use_proba": self.use_proba,
            "passthrough": self.passthrough,
        }

    def set_params(self, **params: Any) -> "StackingEnsemble":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        names = [type(m).__name__ for m in self.base_models]
        meta = type(self.meta_learner).__name__
        return (
            f"StackingEnsemble(base={names}, meta={meta}, "
            f"n_folds={self.n_folds})"
        )


# ============================================================================
# 3. Odds Blender
# ============================================================================

class OddsBlender:
    """Blend model probabilities with odds-implied probabilities.

    The blended probability for home win is::

        p_blend = w * p_model + (1 - w) * p_odds

    where *w* is optimised on a calibration set to minimise log-loss.

    Parameters
    ----------
    model : estimator
        A fitted (or to-be-fitted) model with ``predict_proba``.
    odds_col_home : str
        Column name for home decimal odds in the input DataFrame.
    odds_col_away : str
        Column name for away decimal odds in the input DataFrame.
    weight : float or None
        Fixed blending weight.  If ``None``, the weight is optimised
        during ``fit``.
    refit_model : bool
        If ``True``, ``fit`` also trains the underlying model.
    """

    def __init__(
        self,
        model: Any,
        odds_col_home: str = "home_odds",
        odds_col_away: str = "away_odds",
        weight: float | None = None,
        refit_model: bool = True,
    ) -> None:
        self.model = model
        self.odds_col_home = odds_col_home
        self.odds_col_away = odds_col_away
        self.weight = weight
        self.refit_model = refit_model

        self.optimised_weight_: float | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    @staticmethod
    def _odds_to_home_prob(
        home_odds: np.ndarray,
        away_odds: np.ndarray,
    ) -> np.ndarray:
        """Normalised implied home-win probability from decimal odds."""
        raw_home = 1.0 / np.clip(home_odds, 1.01, None)
        raw_away = 1.0 / np.clip(away_odds, 1.01, None)
        return raw_home / (raw_home + raw_away)

    # ------------------------------------------------------------------
    def _get_odds_prob(self, X: pd.DataFrame) -> np.ndarray:
        """Extract odds-implied home probabilities from the DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "OddsBlender requires a pandas DataFrame with odds columns."
            )
        missing = {self.odds_col_home, self.odds_col_away} - set(X.columns)
        if missing:
            raise ValueError(f"Missing odds columns: {sorted(missing)}")

        return self._odds_to_home_prob(
            X[self.odds_col_home].values.astype(float),
            X[self.odds_col_away].values.astype(float),
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike,
    ) -> "OddsBlender":
        """Fit the underlying model and optimise blending weight.

        Parameters
        ----------
        X : pd.DataFrame
            Training features (must include odds columns).
        y : array-like
            Binary target.

        Returns
        -------
        self
        """
        y_np = np.asarray(y, dtype=int)

        if self.refit_model:
            logger.info("OddsBlender: fitting base model.")
            self.model.fit(X, y_np)

        # If weight is preset, skip optimisation.
        if self.weight is not None:
            self.optimised_weight_ = self.weight
            self.is_fitted_ = True
            return self

        # Optimise weight on the training data.
        p_model = self.model.predict_proba(X)[:, 1]
        p_odds = self._get_odds_prob(X)

        def neg_log_loss(w: float) -> float:
            p_blend = np.clip(
                w * p_model + (1.0 - w) * p_odds, 1e-15, 1.0 - 1e-15
            )
            ll = -(
                y_np * np.log(p_blend)
                + (1 - y_np) * np.log(1.0 - p_blend)
            ).mean()
            return ll

        result = minimize_scalar(
            neg_log_loss, bounds=(0.0, 1.0), method="bounded"
        )
        self.optimised_weight_ = float(result.x)

        logger.info(
            "OddsBlender: optimised weight = %.4f (model) / %.4f (odds).",
            self.optimised_weight_,
            1.0 - self.optimised_weight_,
        )

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return blended class probabilities.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            ``[P(away_win), P(home_win)]``.
        """
        if not self.is_fitted_:
            raise RuntimeError("OddsBlender has not been fitted yet.")

        w = self.optimised_weight_
        p_model = self.model.predict_proba(X)[:, 1]
        p_odds = self._get_odds_prob(X)

        p_home = np.clip(
            w * p_model + (1.0 - w) * p_odds, 1e-15, 1.0 - 1e-15
        )
        return np.column_stack([1.0 - p_home, p_home])

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        return {
            "odds_col_home": self.odds_col_home,
            "odds_col_away": self.odds_col_away,
            "weight": self.weight,
            "refit_model": self.refit_model,
            "optimised_weight": self.optimised_weight_,
        }

    def set_params(self, **params: Any) -> "OddsBlender":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        w_str = (
            f"{self.optimised_weight_:.3f}"
            if self.optimised_weight_ is not None
            else "not optimised"
        )
        return (
            f"OddsBlender(model={type(self.model).__name__}, weight={w_str})"
        )
