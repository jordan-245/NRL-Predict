"""
Scrape season-level team statistics from the NRL.com stats API.

The NRL.com /stats/teams/data endpoint provides per-team season averages
and totals for 33 stat categories across 2013-2025. These are used as
rolling prior-season features for prediction.

Data is saved to data/processed/team_season_stats.parquet.

Schema: year, team, stat_name, total, average
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NRL_API_BASE = "https://www.nrl.com"
TEAM_STATS_API = f"{NRL_API_BASE}/stats/teams/data"
COMPETITION_ID = 111

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "team_season_stats.parquet"

# Team name mapping
TEAM_MAP = {
    "Broncos": "Brisbane Broncos",
    "Raiders": "Canberra Raiders",
    "Bulldogs": "Canterbury Bulldogs",
    "Sharks": "Cronulla Sharks",
    "Titans": "Gold Coast Titans",
    "Sea Eagles": "Manly Sea Eagles",
    "Storm": "Melbourne Storm",
    "Knights": "Newcastle Knights",
    "Cowboys": "North Queensland Cowboys",
    "Eels": "Parramatta Eels",
    "Panthers": "Penrith Panthers",
    "Rabbitohs": "South Sydney Rabbitohs",
    "Roosters": "Sydney Roosters",
    "Dragons": "St George Illawarra Dragons",
    "Warriors": "New Zealand Warriors",
    "NZ Warriors": "New Zealand Warriors",
    "Wests Tigers": "Wests Tigers",
    "Tigers": "Wests Tigers",
    "Dolphins": "Dolphins",
}

# Core stats we want (name -> our column name)
# These are the most predictive for match outcomes
CORE_STATS = {
    "Points": "points",
    "Tries": "tries",
    "Possession %": "possession_pct",
    "Set Completion %": "set_completion_pct",
    "Linebreaks": "line_breaks",
    "Tackle Breaks": "tackle_breaks",
    "Post Contact Metres": "post_contact_metres",
    "All Run Metres": "all_run_metres",
    "All Runs": "all_runs",
    "Kick Return Metres": "kick_return_metres",
    "Offloads": "offloads",
    "Try Assists": "try_assists",
    "Linebreak Assists": "line_break_assists",
    "All Receipts": "all_receipts",
    "Tackles": "tackles",
    "Missed Tackles": "missed_tackles",
    "Ineffective Tackles": "ineffective_tackles",
    "Intercepts": "intercepts",
    "All Kicks": "all_kicks",
    "Kick Metres": "kick_metres",
    "40/20": "forty_twenty",
    "Errors": "errors",
    "Penalties Conceded": "penalties_conceded",
    "Handling Errors": "handling_errors",
    "Goals": "goals",
    "Conversion %": "conversion_pct",
    "Line Engaged": "line_engaged",
    "Supports": "supports",
    "Dummy Half Runs": "dummy_half_runs",
    "Decoy Runs": "decoy_runs",
}


def _standardise_team(nickname: str) -> str:
    return TEAM_MAP.get(nickname, nickname)


def fetch_stat_for_season(
    season: int,
    stat_id: int,
    stat_name: str,
) -> list[dict[str, Any]]:
    """Fetch a single stat for all teams in a season."""
    params = {
        "competition": COMPETITION_ID,
        "season": season,
        "stat": stat_id,
    }
    try:
        r = requests.get(TEAM_STATS_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("Failed to fetch stat %s for %d: %s", stat_name, season, e)
        return []

    rows = []
    for key, field in [("totalStats", "total"), ("averageStats", "average")]:
        section = data.get(key, {})
        leaders = section.get("leaders", [])
        for l in leaders:
            team = _standardise_team(l.get("teamNickName", ""))
            val = l.get("value")
            if team and val is not None:
                try:
                    fval = float(str(val).replace(",", ""))
                except (ValueError, TypeError):
                    continue
                rows.append({
                    "year": season,
                    "team": team,
                    "stat_name": stat_name,
                    "metric": field,
                    "value": fval,
                })

    return rows


def fetch_all_team_stats(
    start_year: int = 2013,
    end_year: int = 2025,
    delay: float = 0.2,
) -> pd.DataFrame:
    """Fetch all team stats for all seasons.

    Returns a wide-format DataFrame: year, team, stat1_avg, stat1_total, ...
    """
    # First, get the stat ID list from the API
    r = requests.get(
        TEAM_STATS_API,
        params={"competition": COMPETITION_ID, "season": end_year, "stat": 30},
        headers=HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()

    stat_list = []
    for s in data.get("filterStats", []):
        name = s.get("name", "")
        sid = s.get("value")
        if name in CORE_STATS and sid is not None:
            stat_list.append((sid, name))

    print(f"  Found {len(stat_list)} core stats to fetch")

    all_rows = []
    for year in range(start_year, end_year + 1):
        for stat_id, stat_name in stat_list:
            rows = fetch_stat_for_season(year, stat_id, stat_name)
            all_rows.extend(rows)
            time.sleep(delay)

        n_teams = len(set(r["team"] for r in all_rows if r["year"] == year))
        print(f"    {year}: {n_teams} teams")

    if not all_rows:
        return pd.DataFrame()

    # Pivot to wide format
    df = pd.DataFrame(all_rows)

    # Create column name from stat_name + metric
    df["col"] = df["stat_name"].map(CORE_STATS).fillna(df["stat_name"]) + "_" + df["metric"]

    # Pivot
    wide = df.pivot_table(
        index=["year", "team"],
        columns="col",
        values="value",
        aggfunc="first",
    ).reset_index()

    wide.columns.name = None

    # Save
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(STATS_PATH, index=False)
    print(f"\n  Saved {len(wide)} team-season rows to {STATS_PATH}")

    return wide


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scrape NRL team season stats")
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  NRL Team Season Stats Scraper")
    print("=" * 60)
    print(f"\n  Fetching {args.start_year}-{args.end_year}...")

    df = fetch_all_team_stats(
        start_year=args.start_year,
        end_year=args.end_year,
        delay=args.delay,
    )

    if len(df):
        print(f"\n  Shape: {df.shape}")
        stat_cols = [c for c in df.columns if c not in ("year", "team")]
        print(f"  Stats: {len(stat_cols)} columns")
        print(f"  Years: {sorted(df['year'].unique())}")
        print(f"  Teams: {df['team'].nunique()} unique")

        # Show sample
        print(f"\n  Sample (2025 Broncos):")
        sample = df[(df["year"] == 2025) & (df["team"] == "Brisbane Broncos")]
        if len(sample):
            for c in sorted(stat_cols)[:10]:
                print(f"    {c:40s} {sample[c].values[0]:.1f}")


if __name__ == "__main__":
    main()
