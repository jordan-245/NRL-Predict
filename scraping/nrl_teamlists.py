"""
Fetch team lists from the NRL.com official API.

The NRL.com website exposes JSON data endpoints for the draw and match
centre pages. These provide:
- Full player names, IDs, positions, and jersey numbers
- 22-man squads for upcoming matches
- 18-man game-day squads for completed matches

No authentication required. Free API.

Usage:
    from scraping.nrl_teamlists import fetch_round_teamlists, fetch_match_teamlist

    # Get all team lists for a round
    teamlists = fetch_round_teamlists(2026, 1)

    # Get a single match
    teamlist = fetch_match_teamlist("/draw/nrl-premiership/2026/round-1/knights-v-cowboys/")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.team_mappings import standardise_team_name
from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

NRL_BASE = "https://www.nrl.com"
DRAW_API = NRL_BASE + "/draw/data"
COMPETITION_ID = 111  # NRL Premiership

CACHE_DIR = DATA_DIR / "teamlists"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# NRL nickname → canonical team name mapping
NICK_TO_CANONICAL = {
    "knights": "Newcastle Knights",
    "cowboys": "North Queensland Cowboys",
    "bulldogs": "Canterbury Bulldogs",
    "dragons": "St George Illawarra Dragons",
    "storm": "Melbourne Storm",
    "eels": "Parramatta Eels",
    "warriors": "New Zealand Warriors",
    "roosters": "Sydney Roosters",
    "broncos": "Brisbane Broncos",
    "panthers": "Penrith Panthers",
    "sharks": "Cronulla Sharks",
    "titans": "Gold Coast Titans",
    "sea eagles": "Manly Sea Eagles",
    "raiders": "Canberra Raiders",
    "dolphins": "Dolphins",
    "rabbitohs": "South Sydney Rabbitohs",
    "wests tigers": "Wests Tigers",
    "tigers": "Wests Tigers",
}


def _nickname_to_canonical(nickname: str) -> str:
    """Convert NRL.com nickname to canonical team name."""
    key = nickname.strip().lower()
    if key in NICK_TO_CANONICAL:
        return NICK_TO_CANONICAL[key]
    # Fallback: try standardise
    try:
        return standardise_team_name(nickname)
    except KeyError:
        return nickname


def _parse_player(player_data: dict) -> dict:
    """Parse a single player entry from NRL.com API."""
    return {
        "player_id": player_data.get("playerId"),
        "first_name": player_data.get("firstName", ""),
        "last_name": player_data.get("lastName", ""),
        "full_name": f"{player_data.get('firstName', '')} {player_data.get('lastName', '')}".strip(),
        "position": player_data.get("position", ""),
        "jersey_number": player_data.get("number"),
        "is_captain": player_data.get("isCaptain", False),
        "is_on_field": player_data.get("isOnField", True),
    }


def fetch_draw(year: int, round_num: int) -> list[dict]:
    """Fetch the draw/fixtures for a given round from NRL.com API.

    Returns list of fixture dicts with matchCentreUrl for each match.
    """
    params = {
        "competition": COMPETITION_ID,
        "season": year,
        "round": round_num,
    }

    try:
        resp = requests.get(DRAW_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("fixtures", [])
    except Exception as e:
        logger.error("Failed to fetch draw for %d round %d: %s", year, round_num, e)
        return []


def fetch_match_teamlist(match_url: str) -> dict[str, Any] | None:
    """Fetch team list data for a single match from its match centre URL.

    Parameters
    ----------
    match_url : str
        The matchCentreUrl from the draw API, e.g.
        "/draw/nrl-premiership/2026/round-1/knights-v-cowboys/"

    Returns
    -------
    dict with keys: home_team, away_team, home_players, away_players, venue, etc.
    """
    # Build the data endpoint URL
    if match_url.startswith("/"):
        url = NRL_BASE + match_url.rstrip("/") + "/data"
    elif match_url.startswith("http"):
        url = match_url.rstrip("/") + "/data"
    else:
        url = NRL_BASE + "/" + match_url.rstrip("/") + "/data"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Failed to fetch match data from %s: %s", url, e)
        return None

    home_raw = data.get("homeTeam", {})
    away_raw = data.get("awayTeam", {})

    home_players = [_parse_player(p) for p in home_raw.get("players", [])]
    away_players = [_parse_player(p) for p in away_raw.get("players", [])]

    return {
        "match_id": data.get("matchId"),
        "match_state": data.get("matchState"),
        "round_number": data.get("roundNumber"),
        "start_time": data.get("startTime"),
        "venue": data.get("venue"),
        "venue_city": data.get("venueCity"),
        "home_team": _nickname_to_canonical(home_raw.get("nickName", "")),
        "away_team": _nickname_to_canonical(away_raw.get("nickName", "")),
        "home_team_id": home_raw.get("teamId"),
        "away_team_id": away_raw.get("teamId"),
        "home_players": home_players,
        "away_players": away_players,
        "home_starters": [p for p in home_players if p["jersey_number"] and p["jersey_number"] <= 13],
        "away_starters": [p for p in away_players if p["jersey_number"] and p["jersey_number"] <= 13],
        "home_bench": [p for p in home_players if p["jersey_number"] and p["jersey_number"] >= 14],
        "away_bench": [p for p in away_players if p["jersey_number"] and p["jersey_number"] >= 14],
    }


def fetch_round_teamlists(
    year: int,
    round_num: int,
    *,
    use_cache: bool = True,
    delay: float = 1.0,
) -> list[dict]:
    """Fetch team lists for all matches in a round.

    Parameters
    ----------
    year : int
        Season year.
    round_num : int
        Round number.
    use_cache : bool
        If True, return cached data if available.
    delay : float
        Delay between API requests (seconds).

    Returns
    -------
    list of dicts, one per match with full team list data.
    """
    cache_path = CACHE_DIR / f"round_{round_num}_{year}.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        logger.info("Loaded cached team lists for %d round %d (%d matches)",
                     year, round_num, len(cached))
        return cached

    # Step 1: Get fixture URLs from draw API
    fixtures = fetch_draw(year, round_num)
    if not fixtures:
        logger.warning("No fixtures found for %d round %d", year, round_num)
        return []

    # Step 2: Fetch team list for each match
    teamlists = []
    for fix in fixtures:
        match_url = fix.get("matchCentreUrl")
        if not match_url:
            continue

        tl = fetch_match_teamlist(match_url)
        if tl and tl.get("home_players"):
            teamlists.append(tl)
            logger.info("  %s v %s: %d home, %d away players",
                       tl["home_team"], tl["away_team"],
                       len(tl["home_players"]), len(tl["away_players"]))

        if delay > 0:
            time.sleep(delay)

    # Cache results
    if teamlists:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(teamlists, f, indent=2)
        logger.info("Cached %d team lists for %d round %d", len(teamlists), year, round_num)

    return teamlists


def get_expected_starters(
    team: str,
    appearances_df: pd.DataFrame | None = None,
    n_recent: int = 5,
) -> dict[int, str]:
    """Get the expected starting 13 for a team based on recent appearances.

    Returns dict mapping jersey_number → player_name (most frequent starter
    at each position in the last n_recent games).
    """
    import pandas as pd

    if appearances_df is None:
        app_path = Path(PROJECT_ROOT / "data" / "processed" / "player_appearances.parquet")
        if not app_path.exists():
            return {}
        appearances_df = pd.read_parquet(app_path)

    # Filter to team starters
    team_apps = appearances_df[
        (appearances_df["team"] == team) &
        (appearances_df["is_starter"])
    ].copy()

    if team_apps.empty:
        return {}

    # Get the most recent rounds
    team_apps = team_apps.sort_values(["year", "round"], ascending=False)
    recent_matches = team_apps["match_id"].unique()[:n_recent * 13]  # ~13 per match
    team_apps = team_apps[team_apps["match_id"].isin(recent_matches[:n_recent])]

    # Most frequent player at each jersey number
    expected = {}
    for jersey in range(1, 14):
        pos_apps = team_apps[team_apps["jersey_number"] == jersey]
        if not pos_apps.empty:
            most_common = pos_apps["player_name"].mode()
            if not most_common.empty:
                expected[jersey] = most_common.iloc[0]

    return expected


def diff_lineups(
    team: str,
    current_players: list[dict],
    expected_starters: dict[int, str],
) -> list[dict]:
    """Compare current team list against expected starters (legacy).

    This uses historical appearances as the baseline — only suitable for
    mid-season when no baseline teamlist is available.  Prefer
    diff_against_baseline() for pregame checks.

    Returns list of changes: who's out, who's in, at what position.
    """
    changes = []
    current_by_number = {p["jersey_number"]: p for p in current_players if p.get("jersey_number")}

    for jersey_num, expected_name in expected_starters.items():
        if jersey_num > 13:
            continue

        current_player = current_by_number.get(jersey_num)
        if current_player is None:
            changes.append({
                "team": team,
                "jersey_number": jersey_num,
                "expected": expected_name,
                "actual": None,
                "change_type": "MISSING",
            })
        else:
            # Compare surnames (RLP has surnames only, NRL.com has full names)
            expected_surname = expected_name.split()[-1].lower() if expected_name else ""
            actual_surname = current_player.get("last_name", "").lower()

            if expected_surname != actual_surname:
                changes.append({
                    "team": team,
                    "jersey_number": jersey_num,
                    "expected": expected_name,
                    "actual": current_player.get("full_name", "Unknown"),
                    "change_type": "REPLACED",
                })

    return changes


# ── Baseline teamlist management ──────────────────────────────────
# The "baseline" is the team list snapshot taken when tips are first
# generated (typically Tuesday).  Pregame checks should compare game-day
# teamlists against this baseline to detect genuine late scratches, NOT
# against historical appearances from prior seasons.

BASELINE_DIR = CACHE_DIR / "baselines"


def save_baseline(year: int, round_num: int, teamlists: list[dict] | None = None) -> Path:
    """Save the current teamlists as the baseline for pregame diffs.

    If teamlists is None, fetches fresh from NRL.com.
    Called during the Tuesday tips pipeline after predictions are generated.

    Returns the path to the saved baseline file.
    """
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"round_{round_num}_{year}.json"

    if teamlists is None:
        teamlists = fetch_round_teamlists(year, round_num, use_cache=True, delay=0.5)

    if not teamlists:
        logger.warning("No teamlists to save as baseline for %d round %d", year, round_num)
        return path

    # Extract just the starters (1-13) per team for clean comparison
    baseline = {}
    for tl in teamlists:
        for side in ("home", "away"):
            team = tl[f"{side}_team"]
            starters = {}
            for p in tl.get(f"{side}_starters", tl.get(f"{side}_players", [])):
                jn = p.get("jersey_number")
                if jn and jn <= 13:
                    starters[str(jn)] = {
                        "full_name": p.get("full_name", ""),
                        "last_name": p.get("last_name", ""),
                        "player_id": p.get("player_id"),
                        "position": p.get("position", ""),
                    }
            baseline[team] = starters

    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info("Saved baseline teamlists for %d round %d (%d teams)", year, round_num, len(baseline))
    return path


def load_baseline(year: int, round_num: int) -> dict[str, dict] | None:
    """Load the baseline teamlist snapshot for a round.

    Returns dict: {team_name: {jersey_num_str: {full_name, last_name, ...}}}
    or None if no baseline exists.
    """
    path = BASELINE_DIR / f"round_{round_num}_{year}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def diff_against_baseline(
    team: str,
    current_players: list[dict],
    baseline: dict[str, dict],
) -> list[dict]:
    """Compare current NRL.com teamlist against the Tuesday baseline.

    This is the correct way to detect game-day scratches: compare the
    fresh pre-kickoff teamlist against the squad announced earlier in
    the week.  Off-season roster moves don't show up because the
    baseline already reflects the current-season squad.

    Parameters
    ----------
    team : str
        Canonical team name.
    current_players : list[dict]
        Fresh player list from NRL.com (full squad including bench).
    baseline : dict
        Output of load_baseline() — {team: {jersey: player_info}}.

    Returns
    -------
    list of change dicts with: team, jersey_number, expected, actual, change_type
    """
    team_baseline = baseline.get(team, {})
    if not team_baseline:
        return []

    changes = []
    current_by_number = {p["jersey_number"]: p for p in current_players if p.get("jersey_number")}

    for jersey_str, expected_info in team_baseline.items():
        jersey_num = int(jersey_str)
        if jersey_num > 13:
            continue

        expected_name = expected_info.get("full_name", "")
        expected_surname = expected_info.get("last_name", "").lower()
        if not expected_surname and expected_name:
            expected_surname = expected_name.split()[-1].lower()

        current_player = current_by_number.get(jersey_num)
        if current_player is None:
            changes.append({
                "team": team,
                "jersey_number": jersey_num,
                "expected": expected_name,
                "actual": None,
                "change_type": "MISSING",
            })
        else:
            actual_surname = current_player.get("last_name", "").lower()
            if expected_surname and actual_surname and expected_surname != actual_surname:
                changes.append({
                    "team": team,
                    "jersey_number": jersey_num,
                    "expected": expected_name,
                    "actual": current_player.get("full_name", "Unknown"),
                    "change_type": "REPLACED",
                })

    return changes


# Allow import of pandas for get_expected_starters
try:
    import pandas as pd
except ImportError:
    pd = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NRL.com team lists")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"\nFetching team lists for {args.year} Round {args.round}...")
    teamlists = fetch_round_teamlists(
        args.year, args.round,
        use_cache=not args.no_cache,
    )

    for tl in teamlists:
        print(f"\n{'=' * 60}")
        print(f"  {tl['home_team']} v {tl['away_team']}")
        print(f"  Venue: {tl['venue']}")
        print(f"  {'-' * 56}")

        for side in ["home", "away"]:
            team = tl[f"{side}_team"]
            print(f"\n  {team}:")
            for p in tl[f"{side}_players"]:
                starter = "*" if p["jersey_number"] and p["jersey_number"] <= 13 else " "
                captain = "(C)" if p.get("is_captain") else "   "
                print(f"    {starter} #{p['jersey_number']:<3d} "
                      f"{p['full_name']:<25s} {p['position']:<15s} {captain}")

    print(f"\n  Total: {len(teamlists)} matches")
