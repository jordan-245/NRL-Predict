"""
Opponent-Adjusted Rolling Stats
================================
Adjusts a team's recent match stats by the quality of opposition faced.

A team posting high completion rates against bottom-4 teams is less impressive
than the same completion rate against top-4 teams.  We weight each stat value
by the opponent's Elo rating relative to the league mean for that match.

Features added (5 stats × 3 columns = 15):
  - home_oa_{stat}_5  / away_oa_{stat}_5  / oa_diff_{stat}_5
  for stat in [completion_rate, line_breaks, errors, all_run_metres, missed_tackles]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Stats to opponent-adjust (top 5 most predictive process stats)
OA_STATS = ["completion_rate", "line_breaks", "errors", "all_run_metres", "missed_tackles"]
OA_WINDOW = 5


def compute_opponent_adjusted_features(
    matches: pd.DataFrame,
    match_stats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute opponent-quality-adjusted rolling stats.

    For each team in each match, takes the last 5 games' stats and weights
    each value by (opponent_elo / league_mean_elo).  Playing well against
    strong opponents lifts the adjusted stat; padding stats against weak
    opponents discounts it.

    Parameters
    ----------
    matches : pd.DataFrame
        Main matches DataFrame with Elo columns already computed.
    match_stats_df : pd.DataFrame or None
        Per-game match stats from match_stats.parquet.

    Returns
    -------
    pd.DataFrame
        matches with 15 new opponent-adjusted feature columns.
    """
    print("\n" + "=" * 80)
    print("  V4.1: COMPUTING OPPONENT-ADJUSTED ROLLING STATS")
    print("=" * 80)

    df = matches.copy().reset_index(drop=True)

    # Initialise output columns to NaN
    for stat in OA_STATS:
        df[f"home_oa_{stat}_{OA_WINDOW}"] = np.nan
        df[f"away_oa_{stat}_{OA_WINDOW}"] = np.nan
        df[f"oa_diff_{stat}_{OA_WINDOW}"] = np.nan

    if match_stats_df is None or len(match_stats_df) == 0:
        print("  WARNING: match_stats_df is None or empty — skipping opponent-adjusted features")
        return df

    ms = match_stats_df.copy()

    # Check available stats
    available = [s for s in OA_STATS if f"home_{s}" in ms.columns and f"away_{s}" in ms.columns]
    if not available:
        print("  WARNING: No matching stat columns — skipping")
        return df

    # Normalise join keys
    ms["year"] = pd.to_numeric(ms["year"], errors="coerce").astype("Int64")
    ms["_round_str"] = ms["round"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["_round_str"] = df["round"].astype(str)
    df["_idx"] = range(len(df))

    # Merge match_stats with matches to get Elo + date for each game
    ms_slim = ms[["year", "_round_str", "home_team", "away_team"]
                 + [f"home_{s}" for s in available]
                 + [f"away_{s}" for s in available]].copy()

    ref = df[["_idx", "year", "_round_str", "home_team", "away_team",
              "date", "home_elo", "away_elo"]].copy()

    joined = ms_slim.merge(ref, on=["year", "_round_str", "home_team", "away_team"], how="inner")

    if len(joined) == 0:
        print("  WARNING: No matches joined — skipping")
        df.drop(columns=["_round_str", "_idx"], errors="ignore", inplace=True)
        return df

    # Compute league mean Elo per season (for normalisation)
    season_mean_elo: dict[int, float] = {}
    for yr in df["year"].dropna().unique():
        yr_mask = df["year"] == yr
        elos = pd.concat([df.loc[yr_mask, "home_elo"], df.loc[yr_mask, "away_elo"]]).dropna()
        season_mean_elo[int(yr)] = float(elos.mean()) if len(elos) > 0 else 1500.0

    # Build per-team match log: each row = one team's stats + opponent Elo weight
    records = []
    for _, row in joined.iterrows():
        yr = int(row["year"]) if pd.notna(row["year"]) else 0
        mean_elo = season_mean_elo.get(yr, 1500.0)

        # Home team's stats, weighted by away team's Elo
        opp_elo_h = row["away_elo"] if pd.notna(row["away_elo"]) else mean_elo
        weight_h = opp_elo_h / mean_elo if mean_elo > 0 else 1.0

        # Away team's stats, weighted by home team's Elo
        opp_elo_a = row["home_elo"] if pd.notna(row["home_elo"]) else mean_elo
        weight_a = opp_elo_a / mean_elo if mean_elo > 0 else 1.0

        base = {"idx": int(row["_idx"]), "date": row["date"]}

        h_rec = {**base, "team": row["home_team"], "weight": weight_h}
        a_rec = {**base, "team": row["away_team"], "weight": weight_a}

        for stat in available:
            h_rec[stat] = row.get(f"home_{stat}", np.nan)
            a_rec[stat] = row.get(f"away_{stat}", np.nan)

        records.append(h_rec)
        records.append(a_rec)

    team_log = pd.DataFrame(records)
    team_log = team_log.sort_values(["team", "date", "idx"]).reset_index(drop=True)

    # Build lookup: (team, match_idx) → {oa_stat_W: weighted_mean}
    lookup: dict[tuple, dict] = {}
    for team in team_log["team"].unique():
        t_log = team_log[team_log["team"] == team].reset_index(drop=True)
        for i, row in t_log.iterrows():
            midx = int(row["idx"])
            prior = t_log.iloc[max(0, i - OA_WINDOW):i]
            if len(prior) == 0:
                lookup[(team, midx)] = {f"{s}_{OA_WINDOW}": np.nan for s in available}
                continue

            entry = {}
            for stat in available:
                vals = prior[stat].values
                weights = prior["weight"].values
                valid = ~np.isnan(vals) & ~np.isnan(weights)
                if valid.sum() > 0:
                    weighted_sum = np.sum(vals[valid] * weights[valid])
                    weight_sum = np.sum(weights[valid])
                    entry[f"{stat}_{OA_WINDOW}"] = weighted_sum / weight_sum if weight_sum > 0 else np.nan
                else:
                    entry[f"{stat}_{OA_WINDOW}"] = np.nan
            lookup[(team, midx)] = entry

    # Map to DataFrame
    for stat in available:
        col_h = f"home_oa_{stat}_{OA_WINDOW}"
        col_a = f"away_oa_{stat}_{OA_WINDOW}"
        col_d = f"oa_diff_{stat}_{OA_WINDOW}"

        df[col_h] = [lookup.get((df.at[i, "home_team"], i), {}).get(f"{stat}_{OA_WINDOW}", np.nan)
                     for i in range(len(df))]
        df[col_a] = [lookup.get((df.at[i, "away_team"], i), {}).get(f"{stat}_{OA_WINDOW}", np.nan)
                     for i in range(len(df))]
        df[col_d] = df[col_h] - df[col_a]

    coverage = df[f"home_oa_{available[0]}_{OA_WINDOW}"].notna().mean() * 100
    print(f"  Added {len(available) * 3} opponent-adjusted features "
          f"({len(available)} stats × {OA_WINDOW}-game window) | Coverage: {coverage:.0f}%")

    df.drop(columns=["_round_str", "_idx"], errors="ignore", inplace=True)
    return df
