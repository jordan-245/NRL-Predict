"""
Enhanced Odds Movement Features (V4.2)
======================================
Probability-space movement features derived from open→close odds shifts.
Complements existing V4 spread_movement/total_movement with probability-level signals.

Features produced (6 total):
  - prob_shift         : implied_prob_close - implied_prob_open (signed)
  - prob_shift_abs     : |prob_shift| (movement magnitude)
  - sharp_money_flag   : |prob_shift| > 0.05 (37% of games — strong signal)
  - favourite_drift    : positive = favourite became more favoured (signed)
  - closing_range_pct  : (close - min) / (max - min) for home odds (0=closed at low, 1=at high)
  - line_overreaction  : |spread_close - spread_open| / |spread_open| (relative movement)

These capture information the existing spread_movement misses:
  - Spread movement is in points; prob_shift is in probability space (more comparable across games)
  - Sharp money flag identifies games where bookmakers received significant late action
  - Favourite drift is directional (did the favourite get shorter or longer?)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_odds_movement_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute probability-space odds movement features.

    Parameters
    ----------
    matches : pd.DataFrame
        Must contain odds columns: h2h_home_open, h2h_home_close,
        h2h_home_min, h2h_home_max, line_home_open, line_home_close.

    Returns
    -------
    pd.DataFrame
        Input with 6 new columns appended.
    """
    print("\n" + "=" * 80)
    print("  V4.2: COMPUTING ENHANCED ODDS MOVEMENT FEATURES")
    print("=" * 80)

    df = matches.copy()
    n = len(df)

    # ── Convert open/close odds to implied probabilities ──────────────────
    h2h_open = pd.to_numeric(df.get("h2h_home_open", pd.Series(dtype=float)), errors="coerce")
    h2h_close = pd.to_numeric(df.get("h2h_home_close", pd.Series(dtype=float)), errors="coerce")
    h2h_away_open = pd.to_numeric(df.get("h2h_away_open", pd.Series(dtype=float)), errors="coerce")
    h2h_min = pd.to_numeric(df.get("h2h_home_min", pd.Series(dtype=float)), errors="coerce")
    h2h_max = pd.to_numeric(df.get("h2h_home_max", pd.Series(dtype=float)), errors="coerce")

    prob_open = 1.0 / h2h_open
    prob_close = 1.0 / h2h_close

    # ── 1. Probability shift (signed: positive = home became more favoured) ──
    df["prob_shift"] = prob_close - prob_open
    valid_shift = df["prob_shift"].notna().sum()

    # ── 2. Absolute probability shift (movement magnitude) ───────────────
    df["prob_shift_abs"] = df["prob_shift"].abs()

    # ── 3. Sharp money flag (|shift| > 5% — captures 37% of games) ──────
    df["sharp_money_flag"] = (df["prob_shift_abs"] > 0.05).astype(float)
    df.loc[df["prob_shift_abs"].isna(), "sharp_money_flag"] = np.nan

    # ── 4. Favourite drift (signed: +1 = fav got shorter, -1 = got longer) ──
    # If home is favourite (prob_open > 0.5), prob_shift > 0 means fav drifted in
    # If away is favourite (prob_open < 0.5), prob_shift < 0 means fav drifted in
    is_home_fav = prob_open > 0.5
    # Favourite drift: positive = favourite became MORE favoured
    df["favourite_drift"] = np.where(
        is_home_fav,
        df["prob_shift"],       # Home fav: positive shift = more favoured
        -df["prob_shift"],      # Away fav: negative shift = away more favoured → flip sign
    )
    df.loc[prob_open.isna() | prob_close.isna(), "favourite_drift"] = np.nan

    # ── 5. Closing range position (where in min-max range did it close?) ──
    odds_range = h2h_max - h2h_min
    # Avoid division by zero
    safe_range = odds_range.replace(0, np.nan)
    df["closing_range_pct"] = (h2h_close - h2h_min) / safe_range
    # Clip to [0, 1] — rounding errors can push slightly outside
    df["closing_range_pct"] = df["closing_range_pct"].clip(0, 1)

    # ── 6. Line overreaction (relative spread movement) ──────────────────
    line_open = pd.to_numeric(df.get("line_home_open", pd.Series(dtype=float)), errors="coerce")
    line_close = pd.to_numeric(df.get("line_home_close", pd.Series(dtype=float)), errors="coerce")
    safe_line = line_open.abs().replace(0, np.nan)
    df["line_overreaction"] = (line_close - line_open).abs() / safe_line
    # Cap extreme values (some games have tiny opening lines)
    df["line_overreaction"] = df["line_overreaction"].clip(upper=5.0)

    # ── Summary ───────────────────────────────────────────────────────────
    features = ["prob_shift", "prob_shift_abs", "sharp_money_flag",
                "favourite_drift", "closing_range_pct", "line_overreaction"]
    for f in features:
        valid = df[f].notna().sum()
        mean = df[f].mean() if valid > 0 else 0
        print(f"    {f:25s}: {valid:4d}/{n} valid, mean={mean:.4f}")

    sharp = df["sharp_money_flag"].sum()
    print(f"\n  Sharp money games: {sharp:.0f}/{valid_shift} ({sharp/max(valid_shift,1)*100:.1f}%)")

    return df
