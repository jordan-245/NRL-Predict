"""
Early-Season Dampening Features
================================
Addresses early-season overconfidence by quantifying how much current-season
data is available and dampening signals accordingly.

Features produced:
  - season_data_reliability  : min(round_number, 8) / 8 — 0→1 as season progresses
  - elo_confidence           : elo_diff × season_data_reliability (dampened Elo signal)
  - home_form_reliability    : fraction of key rolling form windows that are non-NaN
  - away_form_reliability    : same for away team

No look-ahead bias — all inputs are pre-computed rolling/Elo values from prior games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Rolling-form columns we check for fill status (produced by v3)
_FORM_COLS_HOME = [
    "home_win_rate_3",
    "home_win_rate_5",
    "home_win_rate_8",
    "home_avg_pf_3",
    "home_avg_pf_5",
]
_FORM_COLS_AWAY = [
    "away_win_rate_3",
    "away_win_rate_5",
    "away_win_rate_8",
    "away_avg_pf_3",
    "away_avg_pf_5",
]


def compute_early_season_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute early-season dampening features.

    Parameters
    ----------
    matches : pd.DataFrame
        Main feature DataFrame.  Must contain:
          - round_number  (numeric, added by v3 engineering features)
          - elo_diff      (home_elo - away_elo, added by v3)
          - home_win_rate_*, away_win_rate_*, home_avg_pf_* columns (v3 rolling form)

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new columns appended (same row count).
    """
    print("\n" + "=" * 80)
    print("  V4: COMPUTING EARLY-SEASON DAMPENING FEATURES")
    print("=" * 80)

    df = matches.copy()

    # ── season_data_reliability ───────────────────────────────────────────────
    # Ranges from 0 (round 0) to 1.0 (round 8+)
    rn = pd.to_numeric(df.get("round_number", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["season_data_reliability"] = (rn.clip(upper=8) / 8.0).fillna(0.5)

    # ── elo_confidence ────────────────────────────────────────────────────────
    elo_diff = pd.to_numeric(df.get("elo_diff", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    df["elo_confidence"] = elo_diff * df["season_data_reliability"]

    # ── home_form_reliability / away_form_reliability ─────────────────────────
    # Fraction of key rolling-form columns that are non-NaN.
    # Measures how many windows are actually populated for this match row.
    home_cols_present = [c for c in _FORM_COLS_HOME if c in df.columns]
    away_cols_present = [c for c in _FORM_COLS_AWAY if c in df.columns]

    if home_cols_present:
        df["home_form_reliability"] = df[home_cols_present].notna().mean(axis=1)
    else:
        df["home_form_reliability"] = df["season_data_reliability"]

    if away_cols_present:
        df["away_form_reliability"] = df[away_cols_present].notna().mean(axis=1)
    else:
        df["away_form_reliability"] = df["season_data_reliability"]

    n_new = 4
    coverage = df["season_data_reliability"].notna().mean() * 100
    print(f"  Added {n_new} early-season features ({coverage:.0f}% coverage)")

    return df
