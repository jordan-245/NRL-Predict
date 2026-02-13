"""
Fetch NRL fixtures and odds from The Odds API v4.

Provides functions to retrieve upcoming events, head-to-head odds, and
completed scores for the ``rugbyleague_nrl`` sport key.  Team names
are standardised to canonical form via :mod:`config.team_mappings`.

The API key is read from the ``ODDS_API_KEY`` environment variable
(loaded from ``.env``).

Public API
----------
- :func:`get_events`         -- upcoming fixtures (FREE, 0 credits)
- :func:`get_odds`           -- head-to-head odds (1 credit per call)
- :func:`get_scores`         -- completed match scores (2 credits per call)
- :func:`get_upcoming_round` -- fixtures + odds for the next round as a DataFrame

Usage:
    from scraping.odds_api import get_upcoming_round
    df = get_upcoming_round()  # auto-detect round, fetch fixtures + odds
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.team_mappings import standardise_team_name

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com"
SPORT_KEY = "rugbyleague_nrl"

# NRL 2026 season starts ~first week of March.
# This is used for round number estimation.
SEASON_STARTS = {
    2025: datetime(2025, 3, 6, tzinfo=timezone.utc),
    2026: datetime(2026, 3, 5, tzinfo=timezone.utc),
    2027: datetime(2027, 3, 4, tzinfo=timezone.utc),
}


# =====================================================================
# Internal helpers
# =====================================================================

def _get_api_key() -> str:
    """Read ODDS_API_KEY from environment."""
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "ODDS_API_KEY not set. Add it to your .env file:\n"
            "  ODDS_API_KEY=your_key_here"
        )
    return key


def _api_get(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    """Make a GET request to The Odds API and return parsed JSON."""
    url = f"{BASE_URL}{endpoint}"
    if params is None:
        params = {}
    params["apiKey"] = _get_api_key()

    resp = requests.get(url, params=params, timeout=30)

    # Log quota usage from response headers
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if remaining is not None:
        logger.info("API quota: %s used, %s remaining", used, remaining)
        print(f"  [Odds API] Quota: {used} used, {remaining} remaining")

    resp.raise_for_status()
    return resp.json()


def _standardise_api_team(api_name: str) -> str:
    """Map a team name from The Odds API to canonical form."""
    try:
        return standardise_team_name(api_name)
    except KeyError:
        logger.warning(
            "Unrecognised team from Odds API: '%s'. "
            "Add it to TEAM_ALIASES in config/team_mappings.py.",
            api_name,
        )
        return api_name


def _extract_best_odds(event: dict) -> tuple[float | None, float | None]:
    """Extract average home/away h2h odds across all bookmakers.

    Returns (odds_home, odds_away) or (None, None) if no data.
    """
    home_team = event["home_team"]
    away_team = event["away_team"]
    home_prices: list[float] = []
    away_prices: list[float] = []

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market["key"] != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                if outcome["name"] == home_team:
                    home_prices.append(outcome["price"])
                elif outcome["name"] == away_team:
                    away_prices.append(outcome["price"])

    if home_prices and away_prices:
        return (
            sum(home_prices) / len(home_prices),
            sum(away_prices) / len(away_prices),
        )
    return (None, None)


def _extract_spread(event: dict) -> float | None:
    """Extract average home spread (handicap) across all bookmakers.

    Returns the home team's spread (e.g. -5.5 means home favoured by 5.5).
    """
    home_team = event["home_team"]
    spreads: list[float] = []

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market["key"] != "spreads":
                continue
            for outcome in market.get("outcomes", []):
                if outcome["name"] == home_team and "point" in outcome:
                    spreads.append(outcome["point"])

    if spreads:
        return sum(spreads) / len(spreads)
    return None


# =====================================================================
# Public API
# =====================================================================

def get_events() -> list[dict[str, Any]]:
    """Fetch upcoming NRL events (fixtures). FREE — 0 credits.

    Returns list of dicts with: id, commence_time, home_team, away_team.
    Returns empty list during off-season.
    """
    return _api_get(f"/v4/sports/{SPORT_KEY}/events")


def get_odds(
    regions: str = "au",
    markets: str = "h2h",
    odds_format: str = "decimal",
) -> list[dict[str, Any]]:
    """Fetch h2h odds for upcoming NRL events. Costs 1 credit.

    Returns events with nested bookmakers[].markets[].outcomes[].
    """
    return _api_get(
        f"/v4/sports/{SPORT_KEY}/odds/",
        params={
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        },
    )


def get_scores(days_from: int = 3) -> list[dict[str, Any]]:
    """Fetch completed NRL scores. Costs 2 credits.

    Parameters
    ----------
    days_from:
        Number of days back to retrieve scores for.
    """
    return _api_get(
        f"/v4/sports/{SPORT_KEY}/scores/",
        params={"daysFrom": str(days_from)},
    )


def detect_next_round(events: list[dict], year: int = 2026) -> int:
    """Determine the round number from upcoming event dates.

    NRL rounds run roughly weekly (Thu-Mon). We calculate the round
    number based on weeks since the known season start date.
    """
    if not events:
        raise ValueError("No upcoming NRL events found. The season may not have started yet.")

    # Parse earliest event time
    times = []
    for e in events:
        ct = e.get("commence_time")
        if ct:
            t = pd.to_datetime(ct, utc=True)
            times.append(t)

    if not times:
        raise ValueError("No valid commence_time in events.")

    earliest = min(times)

    # Get season start (default to March 1 if year not in lookup)
    season_start = SEASON_STARTS.get(
        year, datetime(year, 3, 1, tzinfo=timezone.utc)
    )
    season_start = pd.Timestamp(season_start)

    days_diff = (earliest - season_start).days
    round_num = max(1, min(27, (days_diff // 7) + 1))

    return round_num


def get_upcoming_round(
    round_num: int | None = None,
    year: int = 2026,
) -> tuple[pd.DataFrame, int]:
    """Fetch fixtures + odds for the next round as a DataFrame.

    This is the primary entry point used by ``predict_round.py --auto``.

    1. Calls get_events() (FREE) for fixtures.
    2. Calls get_odds() (1 credit) for h2h decimal odds.
    3. Merges, standardises team names, detects round.

    Parameters
    ----------
    round_num:
        Explicit round number, or None to auto-detect.
    year:
        Season year.

    Returns
    -------
    (DataFrame, round_num)
        DataFrame with columns matching predict_round.py expectations:
        home_team, away_team, venue, date, year, season, round,
        home_score (NaN), away_score (NaN), h2h_home, h2h_away.
    """
    # Step 1: Get fixtures (FREE)
    print("  Fetching fixtures from Odds API (free)...")
    events = get_events()

    if not events:
        raise ValueError(
            "No upcoming NRL events found.\n"
            "  The season may not have started yet, or it may be the off-season.\n"
            "  Use manual CSV mode instead: python predict_round.py --round N"
        )

    print(f"  Found {len(events)} upcoming events")

    # Step 2: Auto-detect round if needed
    if round_num is None:
        round_num = detect_next_round(events, year)
        print(f"  Auto-detected round: {round_num}")

    # Step 3: Get odds (1 credit for h2h + spreads combined)
    print("  Fetching odds from Odds API (h2h + spreads)...")
    odds_data = get_odds(regions="au", markets="h2h,spreads", odds_format="decimal")

    # Build odds lookup by event ID
    odds_lookup: dict[str, tuple[float | None, float | None]] = {}
    spread_lookup: dict[str, float | None] = {}
    for event in odds_data:
        eid = event["id"]
        odds_lookup[eid] = _extract_best_odds(event)
        spread_lookup[eid] = _extract_spread(event)

    # Step 4: Filter events to next round window
    # NRL rounds run Thu-Mon (~4 days) but opening/split rounds can span
    # up to 9 days.  Use a 10-day window from the earliest event and cap
    # at 8 matches (standard NRL round size).
    times = []
    for e in events:
        ct = e.get("commence_time")
        if ct:
            times.append((pd.to_datetime(ct, utc=True), e))
    times.sort(key=lambda x: x[0])

    if times:
        round_start = times[0][0]
        round_end = round_start + pd.Timedelta(days=10)
        round_events = [e for t, e in times if t <= round_end]
        # Cap at 8 matches (one full NRL round)
        round_events = round_events[:8]
    else:
        round_events = events

    print(f"  Round {round_num}: {len(round_events)} matches")

    # Step 5: Build DataFrame
    rows = []
    for event in round_events:
        home_raw = event["home_team"]
        away_raw = event["away_team"]
        home = _standardise_api_team(home_raw)
        away = _standardise_api_team(away_raw)

        ct = event.get("commence_time")
        date = pd.to_datetime(ct, utc=True) if ct else pd.NaT

        eid = event["id"]
        h2h_home, h2h_away = odds_lookup.get(eid, (None, None))
        home_spread = spread_lookup.get(eid)

        rows.append({
            "home_team": home,
            "away_team": away,
            "venue": "",  # API does not provide venue
            "date": date,
            "year": year,
            "season": year,
            "round": str(round_num),
            "home_score": np.nan,
            "away_score": np.nan,
            "h2h_home": h2h_home,
            "h2h_away": h2h_away,
            "spread_home": home_spread,
        })

    df = pd.DataFrame(rows)

    # Log odds coverage
    has_odds = df["h2h_home"].notna().sum()
    print(f"  Odds available for {has_odds}/{len(df)} matches")

    if has_odds < len(df):
        missing = df[df["h2h_home"].isna()][["home_team", "away_team"]]
        for _, r in missing.iterrows():
            print(f"    No odds: {r['home_team']} vs {r['away_team']}")

    return df, round_num
