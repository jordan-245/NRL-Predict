"""
Player Workload & Origin Depletion Features (V4.2)
===================================================
Team-level workload aggregates from per-player minutes data, plus
State of Origin period indicators.

Data source: data/processed/player_match_stats.parquet (2015+)

Features produced (10 total):
  Workload (per team, home/away/diff):
    - home_starter_mins_avg_3  : Avg minutes per starter over last 3 rounds
    - away_starter_mins_avg_3
    - workload_diff_3          : home - away (fatigue advantage)
    - home_spine_mins_avg_3    : Avg minutes for spine (FB,HB,FE,HK) last 3 rounds
    - away_spine_mins_avg_3
    - spine_workload_diff_3    : home - away spine fatigue

  Consecutive workload:
    - home_heavy_load_count    : # starters with >75 min in ALL of last 3 rounds
    - away_heavy_load_count

  Origin period:
    - is_origin_period         : Round falls in June-July (SOO window)
    - origin_round_number      : 0 outside Origin, 1/2/3 for Origin weeks (cumulative)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _load_player_match_stats() -> pd.DataFrame | None:
    """Load player-level match stats with minutes data."""
    path = PROCESSED_DIR / "player_match_stats.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["year"] = df["year"].astype(int)
    df["round_num"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    return df


def _build_team_workload_history(pms: pd.DataFrame) -> pd.DataFrame:
    """Build per-team-round workload summary from player-level data.

    Returns DataFrame with columns:
        year, round_num, team,
        starter_mins_avg, spine_mins_avg, heavy_load_count
    """
    # Filter to starters only (bench players are rotated, less fatigue signal)
    starters = pms[pms["is_starter"] == True].copy()

    # ── Per-team-round aggregates ──
    team_rounds = (
        starters
        .groupby(["year", "round_num", "team"])
        .agg(
            starter_mins_avg=("minutesPlayed", "mean"),
            starter_count=("minutesPlayed", "count"),
        )
        .reset_index()
    )

    # Spine (FB, HB, FE, HK) minutes
    spine = starters[starters["is_spine"] == True].copy()
    spine_agg = (
        spine
        .groupby(["year", "round_num", "team"])
        .agg(spine_mins_avg=("minutesPlayed", "mean"))
        .reset_index()
    )

    team_rounds = team_rounds.merge(spine_agg, on=["year", "round_num", "team"], how="left")

    # Heavy load count: starters who played >75 min
    starters["is_heavy"] = starters["minutesPlayed"] > 75
    heavy = (
        starters[starters["is_heavy"]]
        .groupby(["year", "round_num", "team"])
        .size()
        .reset_index(name="heavy_count")
    )
    team_rounds = team_rounds.merge(heavy, on=["year", "round_num", "team"], how="left")
    team_rounds["heavy_count"] = team_rounds["heavy_count"].fillna(0).astype(int)

    return team_rounds.sort_values(["team", "year", "round_num"]).reset_index(drop=True)


def _compute_rolling_workload(team_history: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Compute rolling averages of workload stats per team.

    Returns: year, round_num, team, starter_mins_avg_{window}, spine_mins_avg_{window},
             heavy_load_count_{window}
    """
    result_parts = []

    for team, grp in team_history.groupby("team"):
        grp = grp.sort_values(["year", "round_num"]).copy()

        # Rolling mean over last `window` rounds (shift by 1 to avoid leakage)
        grp[f"starter_mins_roll_{window}"] = (
            grp["starter_mins_avg"].shift(1).rolling(window, min_periods=1).mean()
        )
        grp[f"spine_mins_roll_{window}"] = (
            grp["spine_mins_avg"].shift(1).rolling(window, min_periods=1).mean()
        )
        # Heavy load: count of high-minute starters averaged over window
        grp[f"heavy_load_roll_{window}"] = (
            grp["heavy_count"].shift(1).rolling(window, min_periods=1).mean()
        )

        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True)


def _detect_origin_period(matches: pd.DataFrame) -> pd.DataFrame:
    """Add Origin period indicators based on month of match date.

    State of Origin is played in June-July. We mark rounds where ≥50%
    of games fall in June-July as Origin period, then assign cumulative
    Origin round numbers (1, 2, 3) within each season.
    """
    df = matches.copy()

    # Parse dates
    date_col = df.get("parsed_date", df.get("date"))
    dates = pd.to_datetime(date_col, errors="coerce")
    df["_month"] = dates.dt.month

    # Mark individual games in June-July
    df["_in_origin_months"] = df["_month"].isin([6, 7]).astype(int)

    # Determine which year-rounds are "origin period" (majority of games in Jun-Jul)
    round_frac = (
        df.groupby(["year", "round"])["_in_origin_months"]
        .mean()
        .reset_index(name="_origin_frac")
    )
    round_frac["is_origin_period"] = (round_frac["_origin_frac"] >= 0.5).astype(float)

    # Assign cumulative origin round number within each season
    # (Round 1 of Origin = first origin-period round, etc.)
    round_frac["round_num"] = pd.to_numeric(round_frac["round"], errors="coerce")
    origin_rounds = round_frac[round_frac["is_origin_period"] == 1].copy()
    origin_rounds = origin_rounds.sort_values(["year", "round_num"])
    origin_rounds["origin_round_number"] = origin_rounds.groupby("year").cumcount() + 1
    # Cap at 3 (there are only 3 Origin games per year, but window is wider)
    origin_rounds["origin_round_number"] = origin_rounds["origin_round_number"].clip(upper=3)

    # Merge back
    round_frac = round_frac.merge(
        origin_rounds[["year", "round", "origin_round_number"]],
        on=["year", "round"], how="left"
    )
    round_frac["origin_round_number"] = round_frac["origin_round_number"].fillna(0)

    df = df.merge(
        round_frac[["year", "round", "is_origin_period", "origin_round_number"]],
        on=["year", "round"], how="left", suffixes=("", "_orig")
    )

    # Cleanup
    df = df.drop(columns=["_month", "_in_origin_months"], errors="ignore")

    return df


def compute_player_workload_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level player workload and Origin period features.

    Parameters
    ----------
    matches : pd.DataFrame
        Main feature DataFrame. Must contain: year, round, home_team, away_team.

    Returns
    -------
    pd.DataFrame
        Input with 10 new workload/origin columns appended.
    """
    print("\n" + "=" * 80)
    print("  V4.2: COMPUTING PLAYER WORKLOAD + ORIGIN FEATURES")
    print("=" * 80)

    df = matches.copy()
    n = len(df)

    # ── 1. Origin period detection ────────────────────────────────────────
    df = _detect_origin_period(df)
    origin_games = df["is_origin_period"].sum()
    print(f"  Origin period games: {origin_games:.0f}/{n} ({origin_games/n*100:.1f}%)")

    # ── 2. Player workload from match stats ───────────────────────────────
    pms = _load_player_match_stats()
    if pms is None:
        print("  WARNING: player_match_stats.parquet not found — workload features NaN")
        for col in ["home_starter_mins_avg_3", "away_starter_mins_avg_3", "workload_diff_3",
                     "home_spine_mins_avg_3", "away_spine_mins_avg_3", "spine_workload_diff_3",
                     "home_heavy_load_count", "away_heavy_load_count"]:
            df[col] = np.nan
        return df

    # Build team workload history
    team_history = _build_team_workload_history(pms)
    rolling = _compute_rolling_workload(team_history, window=3)

    # Prepare join key
    df["round_num"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df["year"] = df["year"].astype(int)
    rolling["year"] = rolling["year"].astype(int)

    # Join home team workload
    home_cols = {
        f"starter_mins_roll_3": "home_starter_mins_avg_3",
        f"spine_mins_roll_3": "home_spine_mins_avg_3",
        f"heavy_load_roll_3": "home_heavy_load_count",
    }
    home_merge = rolling[["year", "round_num", "team"] + list(home_cols.keys())].copy()
    home_merge = home_merge.rename(columns={**home_cols, "team": "home_team"})
    df = df.merge(home_merge, on=["year", "round_num", "home_team"], how="left")

    # Join away team workload
    away_cols = {
        f"starter_mins_roll_3": "away_starter_mins_avg_3",
        f"spine_mins_roll_3": "away_spine_mins_avg_3",
        f"heavy_load_roll_3": "away_heavy_load_count",
    }
    away_merge = rolling[["year", "round_num", "team"] + list(away_cols.keys())].copy()
    away_merge = away_merge.rename(columns={**away_cols, "team": "away_team"})
    df = df.merge(away_merge, on=["year", "round_num", "away_team"], how="left")

    # ── 3. Compute differentials ──────────────────────────────────────────
    df["workload_diff_3"] = df["home_starter_mins_avg_3"] - df["away_starter_mins_avg_3"]
    df["spine_workload_diff_3"] = df["home_spine_mins_avg_3"] - df["away_spine_mins_avg_3"]

    # Cleanup
    df = df.drop(columns=["round_num"], errors="ignore")

    # ── Summary ───────────────────────────────────────────────────────────
    features = [
        "home_starter_mins_avg_3", "away_starter_mins_avg_3", "workload_diff_3",
        "home_spine_mins_avg_3", "away_spine_mins_avg_3", "spine_workload_diff_3",
        "home_heavy_load_count", "away_heavy_load_count",
        "is_origin_period", "origin_round_number",
    ]
    for f in features:
        if f in df.columns:
            valid = df[f].notna().sum()
            mean = df[f].mean() if valid > 0 else 0
            print(f"    {f:30s}: {valid:4d}/{n} valid, mean={mean:.2f}")

    return df
