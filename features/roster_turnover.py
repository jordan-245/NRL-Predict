"""
Off-Season Roster Turnover Features
=====================================
Measures how many players from the prior year's core squad are still present
in the current season, quantifying off-season continuity / disruption.

Features produced (per match row):
  - home_roster_continuity   : fraction of prior year's core-17 still in R1 squad
  - away_roster_continuity   : same for away team
  - home_spine_continuity    : same, but spine positions only (FB, HB, FE, HK)
  - away_spine_continuity    : same for away team
  - roster_continuity_diff   : home - away roster continuity
  - spine_continuity_diff    : home - away spine continuity

Look-ahead-safe: for a match in year Y, continuity compares:
  - Current squad : Round-1 appearances in year Y
  - Prior core    : Top-17 starters by appearance count in year Y-1

Uses: data/processed/player_appearances.parquet
  columns: year, round, team, player_name, is_starter, is_spine
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Spine positions as stored in player_appearances
SPINE_POSITIONS = {"FB", "HB", "FE", "HK"}


def _load_appearances(appearances_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return the appearances DataFrame, loading from disk if not provided."""
    if appearances_df is not None:
        return appearances_df.copy()
    parquet_path = PROCESSED_DIR / "player_appearances.parquet"
    if not parquet_path.exists():
        return None
    return pd.read_parquet(parquet_path)


def _build_continuity_lookup(
    apps: pd.DataFrame,
) -> dict[tuple[str, int], dict[str, float]]:
    """Pre-compute roster continuity metrics for every (team, year) pair.

    For each (team, year):
      1. r1_squad   = set of players who appeared in Round 1 of that year
      2. prior_core = top-17 starters (by appearance count) in year-1
      3. core_spines = spine members in prior_core

    Returns
    -------
    dict mapping (team, year) → {"roster": float, "spine": float}
    """
    apps = apps.copy()
    apps["year"] = pd.to_numeric(apps["year"], errors="coerce").astype("Int64")

    # ── Round-1 squads per (team, year) ──────────────────────────────────────
    # Use round == '1' or round == 1
    r1_mask = apps["round"].astype(str) == "1"
    r1_starters = apps[r1_mask & apps["is_starter"]]
    r1_squads: dict[tuple[str, int], set[str]] = (
        r1_starters.groupby(["team", "year"])["player_name"]
        .apply(set)
        .to_dict()
    )

    # ── Prior-season core 17 per (team, year) ─────────────────────────────────
    starters = apps[apps["is_starter"]].copy()
    appearance_counts = (
        starters.groupby(["team", "year", "player_name"])
        .size()
        .reset_index(name="app_count")
    )

    prior_core: dict[tuple[str, int], set[str]] = {}
    prior_spine: dict[tuple[str, int], set[str]] = {}

    for (team, year), group in appearance_counts.groupby(["team", "year"]):
        top17 = set(group.nlargest(17, "app_count")["player_name"].tolist())
        prior_core[(team, int(year))] = top17

        # Spine subset: check if player had spine position in that season
        spine_players = set(
            apps[
                apps["is_spine"]
                & (apps["team"] == team)
                & (apps["year"] == year)
            ]["player_name"].tolist()
        )
        prior_spine[(team, int(year))] = top17 & spine_players

    # ── Build lookup ─────────────────────────────────────────────────────────
    lookup: dict[tuple[str, int], dict[str, float]] = {}

    all_teams = apps["team"].unique()
    all_years = sorted(apps["year"].dropna().unique())

    for team in all_teams:
        for year in all_years:
            year_int = int(year)
            prior_year = year_int - 1

            r1_squad = r1_squads.get((team, year_int), set())
            core = prior_core.get((team, prior_year), set())
            spine = prior_spine.get((team, prior_year), set())

            if r1_squad and core:
                roster_cont = len(r1_squad & core) / len(core)
            else:
                roster_cont = np.nan

            if r1_squad and spine:
                spine_cont = len(r1_squad & spine) / len(spine)
            elif core:
                # If no spine found in prior year, fall back to roster continuity
                spine_cont = roster_cont
            else:
                spine_cont = np.nan

            if not np.isnan(roster_cont) or not np.isnan(spine_cont):
                lookup[(team, year_int)] = {
                    "roster": roster_cont,
                    "spine": spine_cont,
                }

    return lookup


def compute_roster_turnover_features(
    matches: pd.DataFrame,
    appearances_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute off-season roster continuity features for each match.

    Parameters
    ----------
    matches : pd.DataFrame
        Main feature DataFrame.  Must contain: home_team, away_team, year.
    appearances_df : pd.DataFrame or None
        player_appearances.parquet data.  If None, loaded from disk.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 6 new columns appended (same row count).
    """
    print("\n" + "=" * 80)
    print("  V4: COMPUTING ROSTER TURNOVER FEATURES")
    print("=" * 80)

    apps = _load_appearances(appearances_df)
    if apps is None or len(apps) == 0:
        print("  WARNING: player_appearances data unavailable — skipping roster features")
        df = matches.copy()
        for col in [
            "home_roster_continuity", "away_roster_continuity",
            "home_spine_continuity",  "away_spine_continuity",
            "roster_continuity_diff", "spine_continuity_diff",
        ]:
            df[col] = np.nan
        return df

    lookup = _build_continuity_lookup(apps)

    df = matches.copy()
    df["_year_int"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    def _get(team_col: str, key: str) -> pd.Series:
        return df.apply(
            lambda row: lookup.get((row[team_col], row["_year_int"]), {}).get(key, np.nan),
            axis=1,
        )

    df["home_roster_continuity"] = _get("home_team", "roster")
    df["away_roster_continuity"] = _get("away_team", "roster")
    df["home_spine_continuity"]  = _get("home_team", "spine")
    df["away_spine_continuity"]  = _get("away_team", "spine")

    df["roster_continuity_diff"] = df["home_roster_continuity"] - df["away_roster_continuity"]
    df["spine_continuity_diff"]  = df["home_spine_continuity"]  - df["away_spine_continuity"]

    df = df.drop(columns=["_year_int"], errors="ignore")

    n_new = 6
    coverage = df["home_roster_continuity"].notna().mean() * 100
    print(f"  Added {n_new} roster continuity features ({coverage:.0f}% coverage)")
    n_teams = len(lookup)
    print(f"  Built continuity lookup for {n_teams} (team, year) entries")

    return df
