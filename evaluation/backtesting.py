"""
Walk-forward backtesting for NRL match-winner prediction models.

Provides :class:`WalkForwardBacktester` which enforces strict temporal splits
so that a model trained on seasons *S_1 .. S_k* is evaluated exclusively on
season *S_{k+1}*.  Random splits are **never** used.

Supports two window modes:

* **Expanding** (default) -- training window grows each fold:
  Train 2013-2017 -> Test 2018, Train 2013-2018 -> Test 2019, ...
* **Sliding** -- only the most recent *N* years are retained in the training
  set.

Models that do not require retraining (e.g. Elo rating systems) are handled
via a ``needs_retraining`` flag on the model factory / model object.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Generator,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import pandas as pd

from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols – lightweight duck-typing contracts for models
# ---------------------------------------------------------------------------


@runtime_checkable
class Predictor(Protocol):
    """Minimal contract for a fitted model."""

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@runtime_checkable
class TrainablePredictor(Protocol):
    """A predictor that exposes a ``fit`` method."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


# Type alias for a callable that produces a fresh (unfitted) model.
ModelFactory = Callable[[], TrainablePredictor]


# ---------------------------------------------------------------------------
# Walk-forward backtester
# ---------------------------------------------------------------------------


class WalkForwardBacktester:
    """Temporal walk-forward backtester for NRL season data.

    Parameters
    ----------
    train_start_year : int
        First year to include in any training window.  Default ``2013``.
    test_years : range | Sequence[int]
        Ordered iterable of years to use as successive test folds.
        Default ``range(2018, 2026)``.
    expanding : bool
        If ``True`` (default) use an expanding training window; if ``False``
        use a sliding window whose width is ``sliding_window_size``.
    sliding_window_size : int
        Number of years in the sliding training window.  Only relevant when
        ``expanding=False``.  Default ``5``.
    year_column : str
        Column name in the DataFrame that contains the season year.
        Default ``"year"``.
    round_column : str
        Column name that contains the round number (used by
        :meth:`run_round_by_round`).  Default ``"round"``.
    """

    def __init__(
        self,
        train_start_year: int = 2013,
        test_years: range | Sequence[int] = range(2018, 2026),
        expanding: bool = True,
        sliding_window_size: int = 5,
        year_column: str = "year",
        round_column: str = "round",
    ) -> None:
        self.train_start_year = train_start_year
        self.test_years = list(test_years)
        self.expanding = expanding
        self.sliding_window_size = sliding_window_size
        self.year_column = year_column
        self.round_column = round_column

        if not self.test_years:
            raise ValueError("test_years must contain at least one year.")
        if sliding_window_size < 1:
            raise ValueError("sliding_window_size must be >= 1.")

    # ------------------------------------------------------------------
    # Split generation
    # ------------------------------------------------------------------

    def get_splits(
        self,
        matches_df: pd.DataFrame,
    ) -> Generator[tuple[np.ndarray, np.ndarray, int], None, None]:
        """Yield ``(train_idx, test_idx, test_year)`` for each test fold.

        Parameters
        ----------
        matches_df : pd.DataFrame
            Must contain ``self.year_column``.

        Yields
        ------
        tuple[np.ndarray, np.ndarray, int]
            Integer-location indices into *matches_df* for the training set,
            the test set, and the test year.
        """
        self._validate_dataframe(matches_df, required_columns=[self.year_column])
        years = matches_df[self.year_column]

        for test_year in self.test_years:
            # Determine training window bounds.
            if self.expanding:
                train_start = self.train_start_year
            else:
                train_start = max(
                    self.train_start_year,
                    test_year - self.sliding_window_size,
                )
            train_end = test_year - 1  # inclusive upper bound

            train_mask = (years >= train_start) & (years <= train_end)
            test_mask = years == test_year

            train_idx = np.flatnonzero(train_mask.values)
            test_idx = np.flatnonzero(test_mask.values)

            if len(train_idx) == 0:
                logger.warning(
                    "No training data for test year %d (train window %d-%d). "
                    "Skipping fold.",
                    test_year,
                    train_start,
                    train_end,
                )
                continue
            if len(test_idx) == 0:
                logger.warning(
                    "No test data for year %d. Skipping fold.",
                    test_year,
                )
                continue

            logger.info(
                "Fold: train %d-%d (%d rows) -> test %d (%d rows)",
                train_start,
                train_end,
                len(train_idx),
                test_year,
                len(test_idx),
            )
            yield train_idx, test_idx, test_year

    # ------------------------------------------------------------------
    # Full backtest run
    # ------------------------------------------------------------------

    def run(
        self,
        model_factory: ModelFactory,
        features_df: pd.DataFrame,
        target: pd.Series,
        metrics_list: Sequence[str] | None = None,
        *,
        needs_retraining: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the walk-forward backtest.

        Parameters
        ----------
        model_factory : callable
            A zero-argument callable that returns an unfitted model instance
            implementing :class:`TrainablePredictor`.
        features_df : pd.DataFrame
            Feature matrix.  Must contain ``self.year_column``.
        target : pd.Series
            Binary target aligned with *features_df*.
        metrics_list : list[str] | None
            Subset of metric names to report. ``None`` reports all.
        needs_retraining : bool
            If ``False``, the model is instantiated once via *model_factory*
            and **not** re-fitted each fold (suitable for Elo-style models
            whose ``predict`` / ``predict_proba`` methods operate on the raw
            features without a separate training step).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            1. Per-year results DataFrame (indexed by test year).
            2. All predictions DataFrame with columns ``year``, ``y_true``,
               ``y_pred``, ``y_prob``.
        """
        self._validate_dataframe(features_df, required_columns=[self.year_column])

        if len(features_df) != len(target):
            raise ValueError(
                f"features_df ({len(features_df)}) and target ({len(target)}) "
                f"must have the same length."
            )

        year_results: list[dict[str, Any]] = []
        all_predictions: list[pd.DataFrame] = []

        # For models that do not need retraining, create once.
        persistent_model: Any | None = None
        if not needs_retraining:
            persistent_model = model_factory()

        feature_cols = [
            c for c in features_df.columns
            if c not in {self.year_column, self.round_column}
        ]

        for train_idx, test_idx, test_year in self.get_splits(features_df):
            X_train = features_df.iloc[train_idx][feature_cols]
            y_train = target.iloc[train_idx]
            X_test = features_df.iloc[test_idx][feature_cols]
            y_test = target.iloc[test_idx]

            if needs_retraining:
                model = model_factory()
                model.fit(X_train, y_train)
            else:
                model = persistent_model

            y_pred = model.predict(X_test)

            # Handle predict_proba output shape (may be (n, 2) or (n,)).
            y_prob_raw = model.predict_proba(X_test)
            y_prob = self._extract_positive_proba(y_prob_raw)

            fold_metrics = compute_all_metrics(y_test, y_pred, y_prob)
            fold_metrics["year"] = test_year
            fold_metrics["n_train"] = len(train_idx)
            fold_metrics["n_test"] = len(test_idx)

            if metrics_list is not None:
                fold_metrics = {
                    k: v
                    for k, v in fold_metrics.items()
                    if k in set(metrics_list) | {"year", "n_train", "n_test"}
                }

            year_results.append(fold_metrics)

            fold_preds = pd.DataFrame(
                {
                    "year": test_year,
                    "y_true": y_test.values,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                }
            )
            all_predictions.append(fold_preds)

        if not year_results:
            logger.warning("No valid folds were produced.")
            return pd.DataFrame(), pd.DataFrame()

        results_df = pd.DataFrame(year_results).set_index("year")
        predictions_df = pd.concat(all_predictions, ignore_index=True)

        # Log aggregate summary.
        logger.info(
            "Backtest complete — aggregate accuracy: %.4f, log-loss: %.4f",
            compute_all_metrics(
                predictions_df["y_true"],
                predictions_df["y_pred"],
                predictions_df["y_prob"],
            )["accuracy"],
            compute_all_metrics(
                predictions_df["y_true"],
                predictions_df["y_pred"],
                predictions_df["y_prob"],
            )["log_loss"],
        )

        return results_df, predictions_df

    # ------------------------------------------------------------------
    # Round-by-round evaluation within a single season
    # ------------------------------------------------------------------

    def run_round_by_round(
        self,
        model: Predictor,
        features_df: pd.DataFrame,
        target: pd.Series,
        year: int,
    ) -> pd.DataFrame:
        """Evaluate an already-fitted model round-by-round within *year*.

        This is useful for analysing how model performance varies across a
        season (e.g., early rounds vs. late rounds).

        Parameters
        ----------
        model : Predictor
            A fitted model with ``predict`` and ``predict_proba`` methods.
        features_df : pd.DataFrame
            Feature matrix.  Must contain ``self.year_column`` and
            ``self.round_column``.
        target : pd.Series
            Binary target aligned with *features_df*.
        year : int
            The season year to evaluate.

        Returns
        -------
        pd.DataFrame
            One row per round with columns for each metric plus ``round``,
            ``n_matches``, ``correct``.
        """
        self._validate_dataframe(
            features_df,
            required_columns=[self.year_column, self.round_column],
        )

        season_mask = features_df[self.year_column] == year
        season_df = features_df.loc[season_mask]
        season_target = target.loc[season_mask]

        if season_df.empty:
            logger.warning("No data found for year %d.", year)
            return pd.DataFrame()

        feature_cols = [
            c for c in features_df.columns
            if c not in {self.year_column, self.round_column}
        ]

        round_results: list[dict[str, Any]] = []

        for rnd, group in season_df.groupby(self.round_column, sort=True):
            idx = group.index
            X_rnd = group[feature_cols]
            y_rnd = season_target.loc[idx]

            if y_rnd.empty:
                continue

            y_pred = model.predict(X_rnd)
            y_prob_raw = model.predict_proba(X_rnd)
            y_prob = self._extract_positive_proba(y_prob_raw)

            n_matches = len(y_rnd)
            correct = int((y_pred == y_rnd.values).sum())

            row: dict[str, Any] = {"round": rnd, "n_matches": n_matches, "correct": correct}

            # Only compute full metrics if we have enough data and both classes.
            if len(np.unique(y_rnd)) >= 2 and n_matches >= 2:
                row.update(compute_all_metrics(y_rnd, y_pred, y_prob))
            else:
                row["accuracy"] = correct / n_matches if n_matches > 0 else float("nan")

            round_results.append(row)

        return pd.DataFrame(round_results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_positive_proba(y_prob_raw: np.ndarray) -> np.ndarray:
        """Extract the probability of the positive class.

        Handles both 1-D arrays (already positive-class proba) and 2-D arrays
        returned by sklearn-style ``predict_proba`` (shape ``(n, 2)``).
        """
        y_prob_raw = np.asarray(y_prob_raw)
        if y_prob_raw.ndim == 2:
            return y_prob_raw[:, 1]
        return y_prob_raw

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
        required_columns: Sequence[str],
    ) -> None:
        """Raise if *df* is missing any required columns."""
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required column(s): {sorted(missing)}. "
                f"Available columns: {list(df.columns)}"
            )
