"""
Elo rating system for NRL teams.

Implements a configurable Elo rating engine with:

* Tunable K-factor, home advantage, initial rating, and season reset
* Three margin-of-victory (MOV) adjustments: none, linear, logarithmic
* Per-season regression toward the mean
* Full history tracking for analysis
* A ``backfill`` convenience method that computes pre-match Elo ratings for
  every row in a historical matches DataFrame -- strictly using only data
  available *before* each match (no leakage).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===================================================================
# Margin-of-victory adjustment modes
# ===================================================================

class MovAdjustment(str, Enum):
    """Margin-of-victory adjustment method."""

    NONE = "none"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"


# ===================================================================
# Rating snapshot (for history)
# ===================================================================

@dataclass(frozen=True)
class RatingSnapshot:
    """Immutable record of a team's rating at a point in time."""

    team: str
    season: int
    round_: str | int
    rating_before: float
    rating_after: float
    opponent: str
    is_home: bool
    score_for: int
    score_against: int


# ===================================================================
# EloRating
# ===================================================================

@dataclass
class EloRating:
    """Configurable Elo rating system for NRL teams.

    Parameters
    ----------
    k_factor:
        Maximum rating change per game.  Higher values make the system more
        reactive.  Default 20.
    home_advantage:
        Rating points added to the home team's effective rating when computing
        expected scores.  Default 50 (roughly 58% expected win rate for
        equal-rated teams at home).
    initial_rating:
        Starting rating for every team.  Default 1500.
    season_reset_factor:
        Between 0 and 1.  At the start of each new season every team's rating
        is regressed toward ``initial_rating`` by this factor:
        ``new = initial + factor * (old - initial)``.  A value of 1.0 means
        no regression; 0.0 means full reset.  Default 0.75.
    mov_adjustment:
        Margin-of-victory adjustment mode.  One of ``"none"``, ``"linear"``,
        ``"logarithmic"``.  Default ``"none"``.
    mov_linear_divisor:
        Divisor for the linear MOV multiplier:
        ``multiplier = 1 + margin / divisor``.  Only used when
        ``mov_adjustment="linear"``.  Default 10.
    """

    k_factor: float = 20.0
    home_advantage: float = 50.0
    initial_rating: float = 1500.0
    season_reset_factor: float = 0.75
    mov_adjustment: MovAdjustment | str = MovAdjustment.NONE
    mov_linear_divisor: float = 10.0

    # Internal state (not constructor parameters)
    _ratings: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _history: List[RatingSnapshot] = field(
        default_factory=list, init=False, repr=False
    )
    _last_season: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Normalise the MOV adjustment to the enum
        if isinstance(self.mov_adjustment, str):
            self.mov_adjustment = MovAdjustment(self.mov_adjustment.lower())

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def get_rating(self, team: str) -> float:
        """Return the current rating for *team* (initialised on first access)."""
        return self._ratings.setdefault(team, self.initial_rating)

    def get_ratings(self) -> Dict[str, float]:
        """Return a copy of the current rating dictionary."""
        return dict(self._ratings)

    def get_history(self) -> List[RatingSnapshot]:
        """Return the full history of rating changes."""
        return list(self._history)

    def get_history_df(self) -> pd.DataFrame:
        """Return the rating history as a DataFrame."""
        if not self._history:
            return pd.DataFrame(
                columns=[
                    "team",
                    "season",
                    "round",
                    "rating_before",
                    "rating_after",
                    "opponent",
                    "is_home",
                    "score_for",
                    "score_against",
                ]
            )
        records = [
            {
                "team": s.team,
                "season": s.season,
                "round": s.round_,
                "rating_before": s.rating_before,
                "rating_after": s.rating_after,
                "opponent": s.opponent,
                "is_home": s.is_home,
                "score_for": s.score_for,
                "score_against": s.score_against,
            }
            for s in self._history
        ]
        return pd.DataFrame(records)

    def reset(self) -> None:
        """Clear all ratings and history."""
        self._ratings.clear()
        self._history.clear()
        self._last_season = None

    # -----------------------------------------------------------------
    # Expected score (win probability)
    # -----------------------------------------------------------------

    def get_expected(self, home_team: str, away_team: str) -> float:
        """Return the expected probability that *home_team* beats *away_team*.

        Incorporates the home-advantage bonus.

        Parameters
        ----------
        home_team:
            Name of the home team.
        away_team:
            Name of the away team.

        Returns
        -------
        float
            Probability in [0, 1] that the home team wins.
        """
        home_r = self.get_rating(home_team) + self.home_advantage
        away_r = self.get_rating(away_team)
        return self._expected(home_r, away_r)

    # -----------------------------------------------------------------
    # Update ratings after a match
    # -----------------------------------------------------------------

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        season: Optional[int] = None,
        round_: Optional[str | int] = None,
    ) -> Tuple[float, float]:
        """Update ratings after a completed match.

        Parameters
        ----------
        home_team, away_team:
            Team names (should already be standardised).
        home_score, away_score:
            Final scores.
        season:
            Season year (used for season-reset logic).
        round_:
            Round identifier (stored in history).

        Returns
        -------
        tuple[float, float]
            (home_new_rating, away_new_rating) after the update.
        """
        # --- Season reset logic -------------------------------------------
        if season is not None:
            self._maybe_season_reset(season)

        home_r_before = self.get_rating(home_team)
        away_r_before = self.get_rating(away_team)

        # Effective ratings (home gets the advantage bonus for expectation)
        home_eff = home_r_before + self.home_advantage
        away_eff = away_r_before

        # Expected scores
        home_exp = self._expected(home_eff, away_eff)
        away_exp = 1.0 - home_exp

        # Actual scores (1 = win, 0.5 = draw, 0 = loss)
        if home_score > away_score:
            home_actual, away_actual = 1.0, 0.0
        elif home_score < away_score:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5

        # Margin-of-victory multiplier
        margin = abs(home_score - away_score)
        mov_mult = self._mov_multiplier(margin)

        # Rating updates
        k_eff = self.k_factor * mov_mult
        home_delta = k_eff * (home_actual - home_exp)
        away_delta = k_eff * (away_actual - away_exp)

        home_r_after = home_r_before + home_delta
        away_r_after = away_r_before + away_delta

        self._ratings[home_team] = home_r_after
        self._ratings[away_team] = away_r_after

        # Record history
        _season = season if season is not None else 0
        _round = round_ if round_ is not None else ""

        self._history.append(
            RatingSnapshot(
                team=home_team,
                season=_season,
                round_=_round,
                rating_before=home_r_before,
                rating_after=home_r_after,
                opponent=away_team,
                is_home=True,
                score_for=home_score,
                score_against=away_score,
            )
        )
        self._history.append(
            RatingSnapshot(
                team=away_team,
                season=_season,
                round_=_round,
                rating_before=away_r_before,
                rating_after=away_r_after,
                opponent=home_team,
                is_home=False,
                score_for=away_score,
                score_against=home_score,
            )
        )

        return home_r_after, away_r_after

    # -----------------------------------------------------------------
    # Backfill: compute Elo for every historical match
    # -----------------------------------------------------------------

    def backfill(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Compute pre-match Elo ratings for every match in a historical table.

        The DataFrame must be sorted chronologically and contain at minimum:
        ``home_team``, ``away_team``, ``home_score``, ``away_score``.
        Optionally ``season`` and ``round`` for season-reset logic and
        history tracking.

        **No data leakage**: each row receives the Elo ratings computed
        strictly from matches *before* it.

        Parameters
        ----------
        matches_df:
            Cleaned, chronologically sorted matches DataFrame.

        Returns
        -------
        pd.DataFrame
            A copy of *matches_df* with three additional columns:

            * ``home_elo`` -- home team's Elo entering the match
            * ``away_elo`` -- away team's Elo entering the match
            * ``home_elo_prob`` -- predicted home-win probability from Elo
        """
        self.reset()

        df = matches_df.copy()
        home_elos: List[float] = []
        away_elos: List[float] = []
        home_probs: List[float] = []

        for _, row in df.iterrows():
            home = row.get("home_team")
            away = row.get("away_team")
            home_score = row.get("home_score")
            away_score = row.get("away_score")
            season = row.get("season")
            round_ = row.get("round")

            if pd.isna(home) or pd.isna(away):
                home_elos.append(np.nan)
                away_elos.append(np.nan)
                home_probs.append(np.nan)
                continue

            # Record PRE-match ratings
            home_r = self.get_rating(home)
            away_r = self.get_rating(away)
            prob = self.get_expected(home, away)

            home_elos.append(home_r)
            away_elos.append(away_r)
            home_probs.append(prob)

            # Update ratings (only if we have valid scores)
            if not (pd.isna(home_score) or pd.isna(away_score)):
                self.update(
                    home_team=home,
                    away_team=away,
                    home_score=int(home_score),
                    away_score=int(away_score),
                    season=int(season) if pd.notna(season) else None,
                    round_=round_ if pd.notna(round_) else None,
                )

        df["home_elo"] = home_elos
        df["away_elo"] = away_elos
        df["home_elo_prob"] = home_probs

        logger.info(
            "backfill: computed Elo for %d matches. "
            "Final rating range: [%.0f, %.0f].",
            len(df),
            min(self._ratings.values()) if self._ratings else 0,
            max(self._ratings.values()) if self._ratings else 0,
        )
        return df

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _expected(rating_a: float, rating_b: float) -> float:
        """Standard Elo expected-score formula."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _mov_multiplier(self, margin: int) -> float:
        """Return the margin-of-victory multiplier for the K-factor."""
        if self.mov_adjustment == MovAdjustment.NONE:
            return 1.0
        elif self.mov_adjustment == MovAdjustment.LINEAR:
            return 1.0 + margin / self.mov_linear_divisor
        elif self.mov_adjustment == MovAdjustment.LOGARITHMIC:
            return math.log(max(margin, 1) + 1)
        else:
            return 1.0

    def _maybe_season_reset(self, season: int) -> None:
        """Regress ratings toward the mean if we've entered a new season."""
        if self._last_season is not None and season > self._last_season:
            factor = self.season_reset_factor
            for team in list(self._ratings):
                old = self._ratings[team]
                self._ratings[team] = (
                    self.initial_rating + factor * (old - self.initial_rating)
                )
            logger.debug(
                "Season reset (%d -> %d) applied with factor %.2f.",
                self._last_season,
                season,
                factor,
            )
        self._last_season = season
