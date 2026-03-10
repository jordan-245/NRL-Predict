"""
Scrape per-player match statistics from the NRL.com API.

For each completed match, fetches the match centre data and extracts
individual player statistics (run metres, line breaks, tackles, etc.)
along with jersey / position metadata.

Public API
----------
- fetch_round_player_stats(year, round_num, delay=0.3)
- fetch_season_player_stats(year)
- backfill_all_player_stats(start_year=2015, end_year=2026)

Data is saved to data/processed/player_match_stats.parquet.

Usage
-----
    python -m scraping.nrl_player_stats                 # backfill 2015-2026
    python -m scraping.nrl_player_stats --year 2026     # full season
    python -m scraping.nrl_player_stats --year 2026 --round 1   # single round
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYER_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "player_match_stats.parquet"
PLAYER_STATS_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "player_stats"

# Jersey number → position code
JERSEY_POSITION_MAP: dict[int, str] = {
    1:  "FB",   # Fullback
    2:  "WG",   # Wing
    3:  "CT",   # Centre
    4:  "CT",   # Centre
    5:  "WG",   # Wing
    6:  "FE",   # Five-Eighth
    7:  "HB",   # Halfback
    8:  "PR",   # Prop
    9:  "HK",   # Hooker (dummy half)
    10: "PR",   # Prop
    11: "SR",   # Second Row
    12: "SR",   # Second Row
    13: "LK",   # Lock
    14: "INT",  # Interchange
    15: "INT",
    16: "INT",
    17: "INT",
}

# Spine positions: FB=1, FE=6, HB=7, HK=9
SPINE_JERSEYS: frozenset[int] = frozenset({1, 6, 7, 9})

# Stat fields to extract from data['stats']['players'] per player entry.
# These are the raw camelCase API field names.
PLAYER_STAT_FIELDS: list[str] = [
    "allRunMetres",
    "allRuns",
    "tacklesMade",
    "tackleBreaks",
    "lineBreaks",
    "lineBreakAssists",
    "tryAssists",
    "tries",
    "fantasyPointsTotal",
    "minutesPlayed",
    "errors",
    "handlingErrors",
    "missedTackles",
    "offloads",
    "postContactMetres",
    "kicks",
    "kickMetres",
    "penalties",
    "ineffectiveTackles",
    "passes",
    "receipts",
    "dummyHalfRuns",
]

# NRL team name standardisation — matches nrl_match_stats.py
TEAM_MAP: dict[str, str] = {
    "Broncos":      "Brisbane Broncos",
    "Raiders":      "Canberra Raiders",
    "Bulldogs":     "Canterbury Bulldogs",
    "Sharks":       "Cronulla Sharks",
    "Titans":       "Gold Coast Titans",
    "Sea Eagles":   "Manly Sea Eagles",
    "Storm":        "Melbourne Storm",
    "Knights":      "Newcastle Knights",
    "Cowboys":      "North Queensland Cowboys",
    "Eels":         "Parramatta Eels",
    "Panthers":     "Penrith Panthers",
    "Rabbitohs":    "South Sydney Rabbitohs",
    "Roosters":     "Sydney Roosters",
    "Dragons":      "St George Illawarra Dragons",
    "Warriors":     "New Zealand Warriors",
    "NZ Warriors":  "New Zealand Warriors",
    "Wests Tigers": "Wests Tigers",
    "Tigers":       "Wests Tigers",
    "Dolphins":     "Dolphins",
}


def _standardise_team(nickname: str) -> str:
    """Map NRL.com nickname to our canonical team name."""
    return TEAM_MAP.get(nickname, nickname)


# ---------------------------------------------------------------------------
# API fetchers (mirror pattern from nrl_match_stats.py)
# ---------------------------------------------------------------------------

def fetch_draw(year: int, round_num: int) -> list[dict]:
    """Fetch the draw/fixture list for a specific round."""
    params = {
        "competition": COMPETITION_ID,
        "season": year,
        "round": round_num,
    }
    try:
        r = requests.get(DRAW_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json().get("fixtures", [])
    except Exception as e:
        logger.warning("Failed to fetch draw for %d R%d: %s", year, round_num, e)
        return []


def fetch_match_data(match_centre_url: str) -> Optional[dict]:
    """Fetch the full match centre JSON from the NRL.com data endpoint.

    Parameters
    ----------
    match_centre_url:
        Path like ``/draw/nrl-premiership/2025/round-1/raiders-v-warriors/``

    Returns
    -------
    dict or None on failure.
    """
    url = f"{NRL_API_BASE}{match_centre_url}data"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Failed to fetch match data from %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _extract_lineup(team_data: dict) -> dict[int, dict]:
    """Extract ``{player_id: {player_name, jersey_number}}`` from a team block.

    The player list is under ``homeTeam.players`` / ``awayTeam.players`` in the
    main match-centre JSON (not in the stats section, which only carries
    ``playerId`` + raw stat values).
    """
    players_info: dict[int, dict] = {}
    for p in team_data.get("players", []):
        pid = p.get("playerId") or p.get("id")
        if pid is None:
            continue
        # NRL API uses firstName + lastName (not fullName)
        first = p.get("firstName") or ""
        last = p.get("lastName") or ""
        name: str = (
            f"{first} {last}".strip()
            or p.get("fullName")
            or p.get("playerName")
            or p.get("name")
            or ""
        )
        jersey_raw = (
            p.get("number")
            or p.get("jerseyNumber")
            or p.get("jersey")
            or 0
        )
        try:
            jersey = int(jersey_raw)
        except (ValueError, TypeError):
            jersey = 0
        players_info[int(pid)] = {"player_name": name, "jersey_number": jersey}
    return players_info


def parse_player_stats(
    match_data: dict,
    year: int,
    round_num: int,
    match_id: str,
) -> list[dict[str, Any]]:
    """Parse a match-centre JSON into a list of per-player stat dicts.

    For each player, merges:
    - Lineup metadata (name, jersey) from ``homeTeam.players`` / ``awayTeam.players``
    - Per-game stats from ``stats.players.homeTeam`` / ``stats.players.awayTeam``

    Returns one flat dict per player with keys:
    ``year, round, match_id, team, player_id, player_name, jersey_number,
    player_position, is_starter, is_spine, <stat_fields>...``
    """
    home_info = match_data.get("homeTeam", {})
    away_info = match_data.get("awayTeam", {})

    home_team = _standardise_team(home_info.get("nickName", ""))
    away_team = _standardise_team(away_info.get("nickName", ""))

    if not home_team or not away_team:
        return []

    # Player lineup metadata (name + jersey) — from main team objects
    home_lineup = _extract_lineup(home_info)
    away_lineup = _extract_lineup(away_info)

    # Per-player stats — from stats.players section
    stats_section = match_data.get("stats", {})
    players_section = stats_section.get("players", {})
    home_player_stats: list[dict] = players_section.get("homeTeam", [])
    away_player_stats: list[dict] = players_section.get("awayTeam", [])

    rows: list[dict[str, Any]] = []

    for team_name, player_stats_list, lineup in [
        (home_team, home_player_stats, home_lineup),
        (away_team, away_player_stats, away_lineup),
    ]:
        for p_stats in player_stats_list:
            pid_raw = p_stats.get("playerId")
            if pid_raw is None:
                continue
            pid = int(pid_raw)

            # Merge with lineup metadata
            info = lineup.get(pid, {})
            player_name: str = info.get("player_name", "")
            jersey: int = info.get("jersey_number", 0)

            position = JERSEY_POSITION_MAP.get(jersey, "INT")
            is_starter = 1 <= jersey <= 13
            is_spine = jersey in SPINE_JERSEYS

            row: dict[str, Any] = {
                "year":            year,
                "round":           str(round_num),
                "match_id":        match_id,
                "team":            team_name,
                "player_id":       pid,
                "player_name":     player_name,
                "jersey_number":   jersey,
                "player_position": position,
                "is_starter":      is_starter,
                "is_spine":        is_spine,
            }

            for field in PLAYER_STAT_FIELDS:
                val = p_stats.get(field)
                if val is not None:
                    try:
                        row[field] = float(val)
                    except (ValueError, TypeError):
                        row[field] = None
                else:
                    row[field] = None

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_round_player_stats(
    year: int,
    round_num: int,
    delay: float = 0.3,
) -> list[dict]:
    """Fetch all player stats for a single round.

    Parameters
    ----------
    year:
        Season year (e.g. 2026).
    round_num:
        Round number (1-based).
    delay:
        Seconds to sleep between API calls.

    Returns
    -------
    List of flat per-player stat dicts (one per player per match).
    """
    fixtures = fetch_draw(year, round_num)
    if not fixtures:
        return []

    rows: list[dict] = []

    for f in fixtures:
        match_state = f.get("matchState", "")
        if match_state not in ("FullTime", "FullTimeExtraTime"):
            continue

        mc_url = f.get("matchCentreUrl")
        if not mc_url:
            continue

        # Use API-provided matchId when available; fall back to URL slug
        match_id = str(
            f.get("matchId") or mc_url.rstrip("/").split("/")[-1]
        )

        match_data = fetch_match_data(mc_url)
        if match_data is None:
            continue

        match_rows = parse_player_stats(match_data, year, round_num, match_id)
        rows.extend(match_rows)

        time.sleep(delay)

    return rows


def fetch_season_player_stats(
    year: int,
    max_rounds: int = 27,
    delay: float = 0.3,
) -> list[dict]:
    """Fetch all player stats for an entire season.

    Stops after two consecutive empty rounds (end of season).

    Parameters
    ----------
    year:
        Season year.
    max_rounds:
        Upper bound on rounds to fetch.
    delay:
        Seconds to sleep between API calls.

    Returns
    -------
    List of flat per-player stat dicts for all rounds.
    """
    all_rows: list[dict] = []
    empty_streak = 0

    for rnd in range(1, max_rounds + 1):
        rows = fetch_round_player_stats(year, rnd, delay=delay)
        if rows:
            all_rows.extend(rows)
            empty_streak = 0
            n_matches = len({r["match_id"] for r in rows})
            print(f"    Round {rnd}: {n_matches} matches, {len(rows)} player entries")
        else:
            empty_streak += 1
            if empty_streak >= 2:
                break

        time.sleep(delay)

    return all_rows


def _cache_path(year: int) -> Path:
    """Per-season JSON cache path."""
    PLAYER_STATS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PLAYER_STATS_CACHE_DIR / f"player_stats_{year}.json"


def backfill_all_player_stats(
    start_year: int = 2015,
    end_year: int = 2026,
    force: bool = False,
    delay: float = 0.3,
) -> pd.DataFrame:
    """Scrape player match stats for all seasons and save to parquet.

    Uses per-season JSON caches to avoid re-fetching completed seasons.

    Parameters
    ----------
    start_year, end_year:
        Inclusive season range.
    force:
        Re-scrape even if a cache file exists.
    delay:
        Seconds between API calls.

    Returns
    -------
    DataFrame with all player match stats (player_match_stats.parquet).
    """
    all_rows: list[dict] = []

    for year in range(start_year, end_year + 1):
        cache = _cache_path(year)

        if cache.exists() and not force:
            with open(cache) as f:
                rows = json.load(f)
            print(f"  {year}: loaded {len(rows)} player entries from cache")
            all_rows.extend(rows)
            continue

        print(f"  {year}: scraping player stats from NRL.com API…")
        rows = fetch_season_player_stats(year, delay=delay)
        print(f"  {year}: {len(rows)} player entries scraped")

        with open(cache, "w") as f:
            json.dump(rows, f)

        all_rows.extend(rows)
        time.sleep(1)  # Courtesy pause between seasons

    if not all_rows:
        print("  No player stats found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["round"] = df["round"].astype(str)

    # Coerce stat columns to numeric
    for field in PLAYER_STAT_FIELDS:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")

    PLAYER_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PLAYER_STATS_PATH, index=False)
    print(f"\n  Saved {len(df)} player stat rows to {PLAYER_STATS_PATH}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape NRL per-player match statistics"
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Season to scrape (default: backfill all)"
    )
    parser.add_argument(
        "--round", type=int, default=None,
        help="Single round to scrape (requires --year)"
    )
    parser.add_argument(
        "--start-year", type=int, default=2015,
        help="Start year for backfill (default: 2015)"
    )
    parser.add_argument(
        "--end-year", type=int, default=2026,
        help="End year for backfill (default: 2026)"
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
    print("  NRL Player Stats Scraper")
    print("=" * 60)

    def _rebuild_parquet() -> None:
        """Merge all per-season JSON caches into a single parquet file."""
        all_rows: list[dict] = []
        for p in sorted(PLAYER_STATS_CACHE_DIR.glob("player_stats_*.json")):
            with open(p) as f:
                all_rows.extend(json.load(f))
        if not all_rows:
            print("  No cached data to rebuild parquet from.")
            return
        df = pd.DataFrame(all_rows)
        df["round"] = df["round"].astype(str)
        for field in PLAYER_STAT_FIELDS:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors="coerce")
        PLAYER_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PLAYER_STATS_PATH, index=False)
        print(f"  Saved {len(df)} total player stat rows to {PLAYER_STATS_PATH}")

    if args.year and args.round is not None:
        # ── Single round ─────────────────────────────────────────────────────
        print(f"\n  Fetching {args.year} Round {args.round}…")
        rows = fetch_round_player_stats(args.year, args.round, delay=args.delay)
        n_matches = len({r["match_id"] for r in rows}) if rows else 0
        print(f"  {n_matches} matches, {len(rows)} player entries")

        if rows:
            cache = _cache_path(args.year)
            existing: list[dict] = []
            if cache.exists() and not args.force:
                with open(cache) as f:
                    existing = json.load(f)
                # Remove any stale entries for this round
                existing = [
                    r for r in existing
                    if str(r.get("round")) != str(args.round)
                ]
            existing.extend(rows)
            with open(cache, "w") as f:
                json.dump(existing, f)
            print(f"  Cache updated: {len(existing)} total entries for {args.year}")
            _rebuild_parquet()

    elif args.year:
        # ── Full season ──────────────────────────────────────────────────────
        print(f"\n  Scraping {args.year} season…")
        rows = fetch_season_player_stats(args.year, delay=args.delay)
        print(f"  {len(rows)} player entries scraped")

        cache = _cache_path(args.year)
        with open(cache, "w") as f:
            json.dump(rows, f)

        _rebuild_parquet()

    else:
        # ── Full backfill ────────────────────────────────────────────────────
        print(f"\n  Backfilling {args.start_year}–{args.end_year}…")
        df = backfill_all_player_stats(
            start_year=args.start_year,
            end_year=args.end_year,
            force=args.force,
            delay=args.delay,
        )
        print(f"\n  Done! {len(df)} total player entries")

        if len(df):
            # Coverage summary
            stat_cols = [c for c in PLAYER_STAT_FIELDS if c in df.columns]
            coverage = df[stat_cols].notna().mean()
            print("\n  Stat coverage (top 10):")
            for col in coverage.sort_values(ascending=False).head(10).index:
                pct = coverage[col] * 100
                print(f"    {col:<30s}  {pct:.0f}%")


if __name__ == "__main__":
    main()
