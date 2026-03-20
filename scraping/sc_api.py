"""
Fetch SuperCoach points-allowed-by-position data from the SC API.

Pulls season-level defensive profiles for every team across all available
seasons (2015-2026) and saves to parquet for use as matchup features.

Output: data/processed/sc_points_allowed.parquet

Usage:
    python -m scraping.sc_api                  # fetch all seasons
    python -m scraping.sc_api --season 2026    # single season
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.team_mappings import standardise_team_name

SC_API_BASE = "http://76.13.193.221"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sc_points_allowed.parquet"

# SC position groups for feature aggregation
SPINE_POSITIONS = {"FLB", "HFB", "FE", "HOK"}
FORWARD_POSITIONS = {"FRF", "2RF"}
BACK_POSITIONS = {"CTW"}


def fetch_teams() -> list[dict]:
    """Fetch team list from SC API."""
    resp = requests.get(f"{SC_API_BASE}/teams", timeout=10)
    resp.raise_for_status()
    return resp.json()["teams"]


def fetch_seasons() -> list[int]:
    """Fetch available seasons from SC API."""
    resp = requests.get(f"{SC_API_BASE}/seasons", timeout=10)
    resp.raise_for_status()
    return resp.json()["seasons"]


def fetch_points_allowed(team_id: int, season: int) -> list[dict] | None:
    """Fetch points-allowed-by-position for a team/season."""
    try:
        resp = requests.get(
            f"{SC_API_BASE}/teams/{team_id}/points-allowed",
            params={"season": season},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("positions", [])
    except (requests.RequestException, KeyError):
        return None


def fetch_all_points_allowed(seasons: list[int] | None = None) -> pd.DataFrame:
    """Fetch points-allowed data for all teams across specified seasons.

    Returns DataFrame with columns:
        season, team, position, avg_points_allowed
    where team uses canonical NRL-Predict names.
    """
    teams = fetch_teams()
    if seasons is None:
        seasons = fetch_seasons()

    records = []
    total = len(teams) * len(seasons)
    done = 0

    for season in sorted(seasons):
        for team in teams:
            done += 1
            positions = fetch_points_allowed(team["team_id"], season)
            if not positions:
                continue

            # Map SC API team name → canonical project name
            try:
                canonical = standardise_team_name(team["team_name"])
            except KeyError:
                canonical = team["team_name"]

            for pos in positions:
                records.append({
                    "season": season,
                    "team": canonical,
                    "position": pos["position"],
                    "avg_points_allowed": pos["avg_points_allowed"],
                })

            if done % 20 == 0:
                print(f"  [{done}/{total}] Fetched {season} {team['team_name']}")

    df = pd.DataFrame(records)
    print(f"  Total records: {len(df)} ({df['season'].nunique()} seasons, "
          f"{df['team'].nunique()} teams)")
    return df


def save(df: pd.DataFrame) -> None:
    """Save to parquet."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved: {OUTPUT_PATH}")


def main():
    print("=" * 60)
    print("  FETCHING SUPERCOACH POINTS-ALLOWED DATA")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()

    seasons = [args.season] if args.season else None
    start = time.time()

    df = fetch_all_points_allowed(seasons)

    if seasons and OUTPUT_PATH.exists():
        # Merge with existing data
        existing = pd.read_parquet(OUTPUT_PATH)
        existing = existing[~existing["season"].isin(seasons)]
        df = pd.concat([existing, df], ignore_index=True)

    save(df)
    print(f"  Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
