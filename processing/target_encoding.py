"""
Target variable creation and target encoding for categorical features.

This module handles:

1. **Target creation** -- deriving the binary ``home_win`` target and the
   continuous ``home_margin`` target from match scores.
2. **Target encoding of categoricals** -- replacing high-cardinality
   categorical features (e.g. venue, team name) with their empirical
   relationship to the target, using proper out-of-fold computation to
   prevent data leakage.

The target-encoding implementation uses K-fold cross-validation so that
each row's encoded value is computed without seeing its own target label.
A global prior (overall target mean) is blended in via a smoothing
parameter to regularise categories with few observations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ===================================================================
# Target creation
# ===================================================================

def create_target(
    matches_df: pd.DataFrame,
    draw_handling: str = "exclude",
) -> pd.Series:
    """Create the binary ``home_win`` target variable.

    Parameters
    ----------
    matches_df:
        Matches DataFrame.  Must contain ``home_score`` and ``away_score``.
    draw_handling:
        How to handle drawn matches.

        * ``"exclude"`` -- return ``NaN`` for draws (downstream code should
          drop those rows before training).
        * ``"home"`` -- treat draws as a home win (1).
        * ``"away"`` -- treat draws as an away win (0).
        * ``"half"`` -- assign 0.5 (only appropriate for regression-style
          targets or custom loss functions).

    Returns
    -------
    pd.Series
        Series of the same length as *matches_df* with values in
        {0, 1, NaN} (or {0, 0.5, 1} if ``draw_handling="half"``).

    Raises
    ------
    ValueError
        If ``draw_handling`` is not one of the recognised options.
    """
    valid_options = {"exclude", "home", "away", "half"}
    if draw_handling not in valid_options:
        raise ValueError(
            f"draw_handling must be one of {valid_options}, got '{draw_handling}'."
        )

    hs = pd.to_numeric(matches_df["home_score"], errors="coerce")
    as_ = pd.to_numeric(matches_df["away_score"], errors="coerce")

    target = pd.Series(np.nan, index=matches_df.index, dtype="Float64", name="home_win")

    home_win_mask = hs > as_
    away_win_mask = as_ > hs
    draw_mask = (hs == as_) & hs.notna()

    target.loc[home_win_mask] = 1.0
    target.loc[away_win_mask] = 0.0

    if draw_handling == "exclude":
        target.loc[draw_mask] = np.nan
    elif draw_handling == "home":
        target.loc[draw_mask] = 1.0
    elif draw_handling == "away":
        target.loc[draw_mask] = 0.0
    elif draw_handling == "half":
        target.loc[draw_mask] = 0.5

    n_home = (target == 1.0).sum()
    n_away = (target == 0.0).sum()
    n_draw = draw_mask.sum()
    n_na = target.isna().sum()
    logger.info(
        "create_target: home_win=%d, away_win=%d, draws=%d, NaN=%d (draw_handling='%s').",
        n_home,
        n_away,
        n_draw,
        n_na,
        draw_handling,
    )
    return target


def create_margin_target(matches_df: pd.DataFrame) -> pd.Series:
    """Create the continuous home-margin target variable.

    ``margin = home_score - away_score``

    Positive values indicate a home win; negative values indicate an away win.

    Parameters
    ----------
    matches_df:
        Matches DataFrame with ``home_score`` and ``away_score``.

    Returns
    -------
    pd.Series
        Series of floats (NaN where scores are missing).
    """
    hs = pd.to_numeric(matches_df["home_score"], errors="coerce")
    as_ = pd.to_numeric(matches_df["away_score"], errors="coerce")
    margin = (hs - as_).rename("home_margin")

    logger.info(
        "create_margin_target: mean=%.1f, std=%.1f, NaN=%d.",
        margin.mean(),
        margin.std(),
        margin.isna().sum(),
    )
    return margin


# ===================================================================
# Target encoding (categorical -> numeric via cross-validation)
# ===================================================================

class TargetEncoder:
    """Target-encode categorical columns with K-fold CV to avoid leakage.

    For **training** data, each row's encoded value is computed from the
    out-of-fold observations, so the row's own target never influences its
    feature.  A Bayesian-style smoothing blends the category mean with the
    global prior to regularise rare categories.

    For **unseen / test** data, the full training-set category means are
    used (with smoothing).

    Parameters
    ----------
    columns:
        List of column names to encode.
    n_folds:
        Number of cross-validation folds.  Default 5.
    smoothing:
        Smoothing strength.  Higher values pull rare-category estimates
        toward the global mean more aggressively.  Default 10.
    random_state:
        Random seed for the fold splitter.  Default 42.

    Attributes
    ----------
    encoding_maps_ : dict[str, dict]
        After ``fit``, maps each column to a dictionary of
        ``{category: encoded_value}``.
    global_mean_ : float
        The overall target mean across the training set.
    """

    def __init__(
        self,
        columns: List[str],
        n_folds: int = 5,
        smoothing: float = 10.0,
        random_state: int = 42,
    ) -> None:
        self.columns = columns
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.random_state = random_state

        # Fitted state
        self.encoding_maps_: Dict[str, Dict] = {}
        self.global_mean_: float = 0.0
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame, target: pd.Series) -> "TargetEncoder":
        """Compute encoding maps from the full training set.

        These maps are used for transforming unseen / test data.  For
        training data, use :meth:`fit_transform` instead (which applies
        cross-validation internally).

        Parameters
        ----------
        df:
            Training DataFrame.
        target:
            Target Series aligned with *df*.

        Returns
        -------
        self
        """
        valid_mask = target.notna()
        df_valid = df.loc[valid_mask]
        target_valid = target.loc[valid_mask]
        self.global_mean_ = float(target_valid.mean())

        for col in self.columns:
            if col not in df_valid.columns:
                logger.warning("TargetEncoder.fit: column '%s' not found; skipping.", col)
                continue

            # Add the target to a temporary frame for groupby
            tmp = df_valid[[col]].copy()
            tmp["__target__"] = target_valid.values
            stats = tmp.groupby(col, dropna=False)["__target__"].agg(["mean", "count"])

            encoding_map = {}
            for cat, row in stats.iterrows():
                cat_mean = row["mean"]
                cat_count = row["count"]
                # Bayesian smoothing
                smoothed = (
                    cat_count * cat_mean + self.smoothing * self.global_mean_
                ) / (cat_count + self.smoothing)
                encoding_map[cat] = smoothed

            self.encoding_maps_[col] = encoding_map

        self._is_fitted = True
        logger.info(
            "TargetEncoder.fit: fitted %d columns, global_mean=%.4f.",
            len(self.encoding_maps_),
            self.global_mean_,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform unseen / test data using the fitted encoding maps.

        Unknown categories are mapped to the global mean.

        Parameters
        ----------
        df:
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with encoded columns replaced.
        """
        if not self._is_fitted:
            raise RuntimeError("TargetEncoder has not been fitted yet.")

        out = df.copy()
        for col, enc_map in self.encoding_maps_.items():
            if col not in out.columns:
                continue
            out[col] = out[col].map(enc_map).fillna(self.global_mean_)
        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        target: pd.Series,
    ) -> pd.DataFrame:
        """Fit on the training set and return cross-validated encoded values.

        Each row's encoded value is computed using only out-of-fold data.
        This prevents the row's own target from leaking into its features.

        Also calls :meth:`fit` internally so that :meth:`transform` can be
        used later on test data.

        Parameters
        ----------
        df:
            Training DataFrame.
        target:
            Target Series aligned with *df*.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with encoded columns replaced.
        """
        # First, fit the full-data maps for later test-time use
        self.fit(df, target)

        out = df.copy()
        valid_mask = target.notna()
        valid_idx = df.index[valid_mask]

        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        for col in self.columns:
            if col not in out.columns:
                continue

            # Initialise with global mean (for rows with NaN targets)
            out[col] = out[col].map(self.encoding_maps_.get(col, {})).fillna(
                self.global_mean_
            )

            # Overwrite valid rows with out-of-fold encoded values
            encoded_vals = pd.Series(np.nan, index=valid_idx, dtype="float64")

            for train_idx, val_idx in kf.split(valid_idx):
                train_positions = valid_idx[train_idx]
                val_positions = valid_idx[val_idx]

                # Compute category stats from training fold
                fold_target = target.loc[train_positions]
                fold_cats = df[col].loc[train_positions]
                fold_global = float(fold_target.mean())

                tmp = pd.DataFrame({"cat": fold_cats, "target": fold_target})
                fold_stats = tmp.groupby("cat", dropna=False)["target"].agg(
                    ["mean", "count"]
                )

                fold_map = {}
                for cat, row in fold_stats.iterrows():
                    cat_mean = row["mean"]
                    cat_count = row["count"]
                    smoothed = (
                        cat_count * cat_mean + self.smoothing * fold_global
                    ) / (cat_count + self.smoothing)
                    fold_map[cat] = smoothed

                # Encode validation fold
                val_cats = df[col].loc[val_positions]
                encoded_vals.loc[val_positions] = val_cats.map(fold_map).fillna(
                    fold_global
                )

            out.loc[valid_idx, col] = encoded_vals.values

        logger.info(
            "TargetEncoder.fit_transform: encoded %d columns via %d-fold CV.",
            len(self.columns),
            self.n_folds,
        )
        return out


# ===================================================================
# Convenience: encode common NRL categoricals
# ===================================================================

def target_encode_categoricals(
    train_df: pd.DataFrame,
    target: pd.Series,
    test_df: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], TargetEncoder]:
    """Convenience wrapper to target-encode categoricals in train (and test).

    Parameters
    ----------
    train_df:
        Training DataFrame.
    target:
        Target Series aligned with *train_df*.
    test_df:
        Optional test DataFrame.  If provided, it is transformed using
        the full training-set encoding maps.
    columns:
        Columns to encode.  If None, a sensible default for NRL data is
        used: ``["venue", "home_team", "away_team"]`` (only those that
        exist in the DataFrame).
    n_folds:
        Number of CV folds.  Default 5.
    smoothing:
        Smoothing parameter.  Default 10.
    random_state:
        Random seed.  Default 42.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None, TargetEncoder]
        (encoded_train, encoded_test_or_None, fitted_encoder)
    """
    if columns is None:
        default_cols = ["venue", "home_team", "away_team"]
        columns = [c for c in default_cols if c in train_df.columns]
        if not columns:
            logger.info("target_encode_categoricals: no categorical columns found.")
            return train_df.copy(), test_df.copy() if test_df is not None else None, TargetEncoder(columns=[])

    encoder = TargetEncoder(
        columns=columns,
        n_folds=n_folds,
        smoothing=smoothing,
        random_state=random_state,
    )

    encoded_train = encoder.fit_transform(train_df, target)

    encoded_test = None
    if test_df is not None:
        encoded_test = encoder.transform(test_df)

    return encoded_train, encoded_test, encoder
