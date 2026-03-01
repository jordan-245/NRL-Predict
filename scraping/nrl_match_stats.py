"""
Scrape detailed match statistics from the NRL.com API.

The NRL.com match centre provides 30+ team-level stats per game including
possession, line breaks, tackle breaks, completion rate, errors, etc.

Public API
----------
- fetch_season_match_stats(year) -- all rounds for a season
- fetch_round_match_stats(year, round_num) -- single round
- backfill_all_stats(start_year, end_year) -- bulk scrape and save

Data is saved to data/processed/match_stats.parquet.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NRL_API_BASE = "https://www.nrl.com"
DRAW_API = f"{NRL_API_BASE}/draw/data"
COMPETITION_ID = 111  # NRL Premiership

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# Where to save the data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "match_stats.parquet"
STATS_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "match_stats"

# Stats we want to extract (stat_title -> column_name)
STAT_COLUMNS = {
    # Possession & Completions
    "Possession %": "possession_pct",
    "Time In Possession": "time_in_possession",
    "Completion Rate": "completion_rate",
    # Attack
    "All Runs": "all_runs",
    "All Run Metres": "all_run_metres",
    "Post Contact Metres": "post_contact_metres",
    "Line Breaks": "line_breaks",
    "Tackle Breaks": "tackle_breaks",
    "Average Set Distance": "avg_set_distance",
    "Kick Return Metres": "kick_return_metres",
    "Average Play The Ball Speed": "avg_ptb_speed",
    # Passing
    "Offloads": "offloads",
    "Receipts": "receipts",
    "Total Passes": "total_passes",
    "Dummy Passes": "dummy_passes",
    # Kicking
    "Kicks": "kicks",
    "Kicking Metres": "kicking_metres",
    "Forced Drop Outs": "forced_dropouts",
    "Kick Defusal %": "kick_defusal_pct",
    "Bombs": "bombs",
    "Grubbers": "grubbers",
    # Defence
    "Effective Tackle %": "effective_tackle_pct",
    "Tackles Made": "tackles_made",
    "Missed Tackles": "missed_tackles",
    "Intercepts": "intercepts",
    "Ineffective Tackles": "ineffective_tackles",
    # Negative Play
    "Errors": "errors",
    "Penalties Conceded": "penalties_conceded",
    "Ruck Infringements": "ruck_infringements",
    "Inside 10 Metres": "inside_10m",
    "On Reports": "on_reports",
    "Sin Bins": "sin_bins",
    # Interchanges
    "Used": "interchanges_used",
}

# NRL team name standardisation (NRL.com nicknames → our canonical names)
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


def _standardise_team(nickname: str) -> str:
    """Map NRL.com nickname to our canonical team name."""
    return TEAM_MAP.get(nickname, nickname)


def _extract_stat_value(stat_entry: dict) -> Optional[float]:
    """Extract numeric value from a stat entry."""
    if isinstance(stat_entry, dict):
        val = stat_entry.get("value")
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


# ---------------------------------------------------------------------------
# API fetchers
# ---------------------------------------------------------------------------

def fetch_draw(year: int, round_num: int) -> list[dict]:
    """Fetch the draw data for a specific round."""
    params = {
        "competition": COMPETITION_ID,
        "season": year,
        "round": round_num,
    }
    try:
        r = requests.get(DRAW_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("fixtures", [])
    except Exception as e:
        logger.warning("Failed to fetch draw for %d R%d: %s", year, round_num, e)
        return []


def fetch_match_stats(match_centre_url: str) -> Optional[dict]:
    """Fetch detailed stats from a match centre page.

    Parameters
    ----------
    match_centre_url:
        Path like /draw/nrl-premiership/2025/round-1/raiders-v-warriors/

    Returns
    -------
    dict with stats groups, or None on failure.
    """
    url = f"{NRL_API_BASE}{match_centre_url}data"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Failed to fetch match stats from %s: %s", url, e)
        return None


def parse_match_stats(
    match_data: dict,
    year: int,
    round_num: int,
) -> Optional[dict[str, Any]]:
    """Parse a match centre response into a flat stats dict.

    Returns a single dict with columns:
    year, round, home_team, away_team, home_score, away_score,
    plus home_* and away_* for each stat in STAT_COLUMNS.
    """
    home_info = match_data.get("homeTeam", {})
    away_info = match_data.get("awayTeam", {})

    home_team = _standardise_team(home_info.get("nickName", ""))
    away_team = _standardise_team(away_info.get("nickName", ""))
    home_score = home_info.get("score")
    away_score = away_info.get("score")

    if not home_team or not away_team:
        return None

    row: dict[str, Any] = {
        "year": year,
        "round": str(round_num),
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
    }

    # Parse stat groups
    stats = match_data.get("stats", {})
    groups = stats.get("groups", [])

    for group in groups:
        for stat in group.get("stats", []):
            title = stat.get("title", "")
            col_name = STAT_COLUMNS.get(title)
            if col_name is None:
                continue

            home_val = _extract_stat_value(stat.get("home"))
            away_val = _extract_stat_value(stat.get("away"))

            row[f"home_{col_name}"] = home_val
            row[f"away_{col_name}"] = away_val

    return row


def fetch_round_match_stats(
    year: int,
    round_num: int,
    delay: float = 0.5,
) -> list[dict]:
    """Fetch all match stats for a single round.

    Returns list of flat stat dicts.
    """
    fixtures = fetch_draw(year, round_num)
    if not fixtures:
        return []

    rows = []
    for f in fixtures:
        match_state = f.get("matchState", "")
        if match_state not in ("FullTime", "FullTimeExtraTime"):
            continue

        mc_url = f.get("matchCentreUrl")
        if not mc_url:
            continue

        match_data = fetch_match_stats(mc_url)
        if match_data is None:
            continue

        row = parse_match_stats(match_data, year, round_num)
        if row:
            rows.append(row)

        time.sleep(delay)

    return rows


def fetch_season_match_stats(
    year: int,
    max_rounds: int = 27,
    delay: float = 0.3,
) -> list[dict]:
    """Fetch all match stats for an entire season.

    Stops when a round returns no data (end of season reached).
    """
    all_rows = []
    empty_streak = 0

    for rnd in range(1, max_rounds + 1):
        rows = fetch_round_match_stats(year, rnd, delay=delay)
        if rows:
            all_rows.extend(rows)
            empty_streak = 0
            print(f"    Round {rnd}: {len(rows)} matches")
        else:
            empty_streak += 1
            if empty_streak >= 2:
                # Two consecutive empty rounds = likely end of season
                break

        time.sleep(delay)

    return all_rows


def _cache_path(year: int) -> Path:
    """Per-season JSON cache file."""
    STATS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return STATS_CACHE_DIR / f"match_stats_{year}.json"


def backfill_all_stats(
    start_year: int = 2015,
    end_year: int = 2025,
    force: bool = False,
    delay: float = 0.3,
) -> pd.DataFrame:
    """Scrape match stats for all seasons and save to parquet.

    Uses per-season JSON caches to avoid re-fetching completed seasons.

    Parameters
    ----------
    start_year, end_year:
        Season range (inclusive).
    force:
        If True, re-scrape even if cache exists.
    delay:
        Seconds between API calls.

    Returns
    -------
    DataFrame with all match stats.
    """
    all_rows = []

    for year in range(start_year, end_year + 1):
        cache = _cache_path(year)

        if cache.exists() and not force:
            with open(cache) as f:
                rows = json.load(f)
            print(f"  {year}: loaded {len(rows)} matches from cache")
            all_rows.extend(rows)
            continue

        print(f"  {year}: scraping from NRL.com API...")
        rows = fetch_season_match_stats(year, delay=delay)
        print(f"  {year}: {len(rows)} matches scraped")

        # Cache
        with open(cache, "w") as f:
            json.dump(rows, f)

        all_rows.extend(rows)
        time.sleep(1)  # Courtesy pause between seasons

    if not all_rows:
        print("  No match stats found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Ensure round is string for consistency
    df["round"] = df["round"].astype(str)

    # Save
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STATS_PATH, index=False)
    print(f"\n  Saved {len(df)} match stats to {STATS_PATH}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scrape NRL match stats")
    parser.add_argument(
        "--year", type=int, default=None,
        help="Single season to scrape (default: backfill all)"
    )
    parser.add_argument(
        "--start-year", type=int, default=2015,
        help="Start year for backfill (default: 2015)"
    )
    parser.add_argument(
        "--end-year", type=int, default=2025,
        help="End year for backfill (default: 2025)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-scrape even if cached"
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Delay between API calls in seconds (default: 0.3)"
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  NRL Match Stats Scraper")
    print("=" * 60)

    if args.year:
        print(f"\n  Scraping {args.year} season...")
        rows = fetch_season_match_stats(args.year, delay=args.delay)
        print(f"  {len(rows)} matches scraped")

        # Save to cache
        cache = _cache_path(args.year)
        with open(cache, "w") as f:
            json.dump(rows, f)

        # Rebuild full parquet from all caches
        all_rows = []
        for p in sorted(STATS_CACHE_DIR.glob("match_stats_*.json")):
            with open(p) as f:
                all_rows.extend(json.load(f))

        if all_rows:
            df = pd.DataFrame(all_rows)
            df["round"] = df["round"].astype(str)
            df.to_parquet(STATS_PATH, index=False)
            print(f"  Saved {len(df)} total match stats to {STATS_PATH}")
    else:
        print(f"\n  Backfilling {args.start_year}-{args.end_year}...")
        df = backfill_all_stats(
            start_year=args.start_year,
            end_year=args.end_year,
            force=args.force,
            delay=args.delay,
        )
        print(f"\n  Done! {len(df)} total matches")

        # Summary
        if len(df):
            stat_cols = [c for c in df.columns if c.startswith("home_") and c != "home_team" and c != "home_score"]
            coverage = df[stat_cols].notna().mean()
            print(f"\n  Stat coverage:")
            for col in sorted(stat_cols, key=lambda c: coverage[c], reverse=True)[:10]:
                pct = coverage[col] * 100
                print(f"    {col.replace('home_', ''):30s} {pct:.0f}%")


if __name__ == "__main__":
    main()
