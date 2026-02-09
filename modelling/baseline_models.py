"""
Baseline prediction models for NRL match winner prediction.

These models establish performance floors that more sophisticated approaches
must beat.  All classes implement a scikit-learn-compatible interface
(``fit``, ``predict``, ``predict_proba``) so they can be used interchangeably
with classical and neural models in the backtesting harness.

Expected columns in *X* (a ``pd.DataFrame``):
    - ``home_team``       : str -- canonical home team name
    - ``away_team``       : str -- canonical away team name
    - ``home_ladder_pos`` : int -- home team ladder position entering the round
    - ``away_ladder_pos`` : int -- away team ladder position entering the round
    - ``home_odds``       : float -- closing head-to-head odds for home team
    - ``away_odds``       : float -- closing head-to-head odds for away team
    - ``season``          : int -- season year (used by EloModel for resets)
    - ``round``           : int | str -- round identifier

Not every model uses every column; missing columns are tolerated when they
are not required by the model in question.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _validate_dataframe(X: Any, required_cols: list[str]) -> pd.DataFrame:
    """Ensure *X* is a DataFrame containing *required_cols*."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(X).__name__}. "
            "Baseline models require named columns."
        )
    missing = set(required_cols) - set(X.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {sorted(missing)}"
        )
    return X


# ============================================================================
# 1. Home Always Wins
# ============================================================================

class HomeAlwaysModel:
    """Always predict the home team to win.

    Historical NRL home-win rate is approximately 55--58 %, so
    ``predict_proba`` returns a fixed ``[0.42, 0.58]`` (away, home)
    for every match, reflecting the base-rate prior.
    """

    _HOME_WIN_PROB: float = 0.58

    def __init__(self) -> None:
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame | ArrayLike,
        y: ArrayLike | None = None,
    ) -> "HomeAlwaysModel":
        """No-op fit (model is parameter-free).

        Parameters
        ----------
        X : array-like
            Ignored.
        y : array-like, optional
            Ignored.

        Returns
        -------
        self
        """
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame | ArrayLike) -> np.ndarray:
        """Predict home win (1) for every match.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data (only the number of rows matters).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Array of ones (home win).
        """
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.ones(n, dtype=int)

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame | ArrayLike) -> np.ndarray:
        """Return fixed probability ``[1 - p, p]`` for every match.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Columns are ``[P(away_win), P(home_win)]``.
        """
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        proba = np.full(
            (n, 2),
            [1.0 - self._HOME_WIN_PROB, self._HOME_WIN_PROB],
        )
        return proba

    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        return {}

    def set_params(self, **params: Any) -> "HomeAlwaysModel":
        return self

    def __repr__(self) -> str:
        return "HomeAlwaysModel()"


# ============================================================================
# 2. Ladder Position Model
# ============================================================================

class LadderModel:
    """Predict the team with the higher (numerically lower) ladder position.

    Requires columns ``home_ladder_pos`` and ``away_ladder_pos`` in *X*.
    When positions are equal the home team is favoured.  Probabilities are
    derived from a simple logistic mapping of the position gap.
    """

    _REQUIRED_COLS: list[str] = ["home_ladder_pos", "away_ladder_pos"]

    def __init__(self, scale: float = 0.15) -> None:
        """
        Parameters
        ----------
        scale : float
            Controls the steepness of the logistic probability curve.
            Larger values produce probabilities closer to 0.5 for small
            position gaps.
        """
        self.scale = scale
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike | None = None,
    ) -> "LadderModel":
        """No-op fit (model is parameter-free beyond *scale*)."""
        _validate_dataframe(X, self._REQUIRED_COLS)
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def _position_gap(self, X: pd.DataFrame) -> np.ndarray:
        """Away ladder position minus home ladder position.

        A positive gap means the home team is ranked higher (better).
        """
        df = _validate_dataframe(X, self._REQUIRED_COLS)
        return (
            df["away_ladder_pos"].values - df["home_ladder_pos"].values
        ).astype(float)

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Logistic mapping of ladder-position gap to home-win probability.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``home_ladder_pos`` and ``away_ladder_pos``.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            ``[P(away_win), P(home_win)]``.
        """
        gap = self._position_gap(X)
        home_prob = 1.0 / (1.0 + np.exp(-self.scale * gap))
        return np.column_stack([1.0 - home_prob, home_prob])

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict 1 (home win) when home team is ranked equal or higher.

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
        return {"scale": self.scale}

    def set_params(self, **params: Any) -> "LadderModel":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return f"LadderModel(scale={self.scale})"


# ============================================================================
# 3. Odds-Implied Probability Model
# ============================================================================

class OddsImpliedModel:
    """Convert closing head-to-head odds to implied probabilities.

    The raw implied probabilities from odds include an overround (bookmaker
    margin).  This model removes the overround via normalisation before
    selecting the favourite.

    Requires columns ``home_odds`` and ``away_odds`` in *X*.
    """

    _REQUIRED_COLS: list[str] = ["home_odds", "away_odds"]

    def __init__(self) -> None:
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike | None = None,
    ) -> "OddsImpliedModel":
        """No-op fit (model is parameter-free)."""
        _validate_dataframe(X, self._REQUIRED_COLS)
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _odds_to_proba(
        home_odds: np.ndarray,
        away_odds: np.ndarray,
    ) -> np.ndarray:
        """Convert decimal odds to normalised probabilities.

        Parameters
        ----------
        home_odds, away_odds : np.ndarray
            Decimal odds (e.g. 1.80, 2.10).

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            ``[P(away_win), P(home_win)]`` after removing overround.
        """
        raw_home = 1.0 / home_odds
        raw_away = 1.0 / away_odds
        total = raw_home + raw_away
        home_prob = raw_home / total
        away_prob = raw_away / total
        return np.column_stack([away_prob, home_prob])

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return odds-implied probabilities for each match.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``home_odds`` and ``away_odds``.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            ``[P(away_win), P(home_win)]``.
        """
        df = _validate_dataframe(X, self._REQUIRED_COLS)
        home_odds = df["home_odds"].values.astype(float)
        away_odds = df["away_odds"].values.astype(float)

        # Guard against zero or negative odds (bad data).
        if np.any(home_odds <= 0) or np.any(away_odds <= 0):
            logger.warning(
                "Encountered non-positive odds; clipping to minimum of 1.01."
            )
            home_odds = np.clip(home_odds, 1.01, None)
            away_odds = np.clip(away_odds, 1.01, None)

        return self._odds_to_proba(home_odds, away_odds)

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict 1 (home win) when implied home probability >= 0.5.

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
        return {}

    def set_params(self, **params: Any) -> "OddsImpliedModel":
        return self

    def __repr__(self) -> str:
        return "OddsImpliedModel()"


# ============================================================================
# 4. Elo Rating Model
# ============================================================================

class EloModel:
    """Predict the winner using an Elo rating system.

    This is a thin wrapper that maintains an internal ``EloRating`` engine
    (from ``processing.elo``) and exposes a scikit-learn-compatible API.

    During ``fit``, it processes all training matches chronologically to
    build up team ratings.  During ``predict`` / ``predict_proba``, it uses
    the *current* ratings snapshot (no updates) to generate predictions.

    Required columns in *X*: ``home_team``, ``away_team``.
    Additionally for ``fit``: ``season`` and the target ``y`` (1 = home win).

    Parameters
    ----------
    k_factor : float
        Elo K-factor controlling update magnitude.
    home_advantage : float
        Fixed Elo points added to the home team's rating.
    initial_rating : float
        Starting rating for teams not yet seen.
    season_reset_factor : float
        Fraction to regress ratings toward the mean between seasons.
        ``1.0`` means no regression; ``0.5`` means halfway reset.
    """

    _REQUIRED_COLS_FIT: list[str] = ["home_team", "away_team", "season"]
    _REQUIRED_COLS_PREDICT: list[str] = ["home_team", "away_team"]

    def __init__(
        self,
        k_factor: float = 25.0,
        home_advantage: float = 50.0,
        initial_rating: float = 1500.0,
        season_reset_factor: float = 0.75,
    ) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.season_reset_factor = season_reset_factor

        # Internal state set during fit.
        self.ratings_: dict[str, float] = {}
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Internal Elo helpers
    # ------------------------------------------------------------------

    def _expected_score(
        self,
        rating_a: float,
        rating_b: float,
    ) -> float:
        """Expected score for player A given both ratings."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _update(
        self,
        home_team: str,
        away_team: str,
        home_win: int,
    ) -> None:
        """Update ratings after a single match."""
        r_home = self.ratings_.get(home_team, self.initial_rating)
        r_away = self.ratings_.get(away_team, self.initial_rating)

        expected_home = self._expected_score(
            r_home + self.home_advantage, r_away
        )
        actual_home = float(home_win)

        self.ratings_[home_team] = r_home + self.k_factor * (
            actual_home - expected_home
        )
        self.ratings_[away_team] = r_away + self.k_factor * (
            (1.0 - actual_home) - (1.0 - expected_home)
        )

    def _apply_season_reset(self) -> None:
        """Regress all ratings toward the mean between seasons."""
        if self.season_reset_factor >= 1.0:
            return
        mean_rating = np.mean(list(self.ratings_.values()))
        for team in self.ratings_:
            self.ratings_[team] = (
                self.season_reset_factor * self.ratings_[team]
                + (1.0 - self.season_reset_factor) * mean_rating
            )

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike,
    ) -> "EloModel":
        """Build Elo ratings from historical match results.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``home_team``, ``away_team``, ``season``.
            Rows should be in chronological order.
        y : array-like of shape (n_samples,)
            Binary target: 1 = home win, 0 = away win.

        Returns
        -------
        self
        """
        df = _validate_dataframe(X, self._REQUIRED_COLS_FIT)
        y_arr = np.asarray(y, dtype=int)

        if len(y_arr) != len(df):
            raise ValueError(
                f"X has {len(df)} rows but y has {len(y_arr)} elements."
            )

        # Reset ratings for a fresh fit.
        self.ratings_ = {}
        current_season: int | None = None

        for idx in range(len(df)):
            row_season = int(df.iloc[idx]["season"])

            # Season boundary: apply reset if we have moved to a new season.
            if current_season is not None and row_season != current_season:
                self._apply_season_reset()
            current_season = row_season

            home_team = str(df.iloc[idx]["home_team"])
            away_team = str(df.iloc[idx]["away_team"])
            self._update(home_team, away_team, int(y_arr[idx]))

        self.is_fitted_ = True
        logger.info(
            "EloModel fitted on %d matches; %d teams rated.",
            len(df),
            len(self.ratings_),
        )
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict home-win probability from current Elo ratings.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``home_team`` and ``away_team``.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            ``[P(away_win), P(home_win)]``.
        """
        if not self.is_fitted_:
            raise RuntimeError("EloModel has not been fitted yet.")

        df = _validate_dataframe(X, self._REQUIRED_COLS_PREDICT)
        n = len(df)
        proba = np.empty((n, 2), dtype=float)

        for idx in range(n):
            home = str(df.iloc[idx]["home_team"])
            away = str(df.iloc[idx]["away_team"])

            r_home = self.ratings_.get(home, self.initial_rating)
            r_away = self.ratings_.get(away, self.initial_rating)

            p_home = self._expected_score(
                r_home + self.home_advantage, r_away
            )
            proba[idx, 0] = 1.0 - p_home
            proba[idx, 1] = p_home

        return proba

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict 1 (home win) when Elo-implied home probability >= 0.5.

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
    def get_ratings(self) -> dict[str, float]:
        """Return a copy of the current ratings dictionary."""
        return dict(self.ratings_)

    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        return {
            "k_factor": self.k_factor,
            "home_advantage": self.home_advantage,
            "initial_rating": self.initial_rating,
            "season_reset_factor": self.season_reset_factor,
        }

    def set_params(self, **params: Any) -> "EloModel":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for EloModel.")
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return (
            f"EloModel(k_factor={self.k_factor}, "
            f"home_advantage={self.home_advantage}, "
            f"initial_rating={self.initial_rating}, "
            f"season_reset_factor={self.season_reset_factor})"
        )
