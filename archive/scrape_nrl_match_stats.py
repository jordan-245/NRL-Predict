"""
Scrape NRL match statistics from NRL.com API and beauhobba S3 bucket.

Collects team-level match stats (completion rate, run metres, tackles, etc.)
for all NRL matches 2013-2025 and saves to data/processed/match_stats.parquet.

Data sources:
  - S3 bulk data (2021-2024): pre-scraped detailed match stats from beauhobba
  - NRL.com match centre API (all other years): team stats per match

Usage:
    python scrape_nrl_match_stats.py
    python scrape_nrl_match_stats.py --years 2023 2024
    python scrape_nrl_match_stats.py --force-rescrape
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DIR, RAW_DIR
from config.team_mappings import standardise_team_name

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scrape_nrl_match_stats")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = RAW_DIR / "nrl_match_stats"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

S3_BULK_URL = (
    "https://geo145327-staging.s3.ap-southeast-2.amazonaws.com"
    "/public/NRL/{year}/NRL_detailed_match_data_{year}.json"
)
S3_YEARS = [2021, 2022, 2023, 2024]

NRL_API_URL = (
    "https://www.nrl.com/draw/nrl-premiership/{year}/{round_slug}/"
    "{home_slug}-v-{away_slug}/data"
)

# Browser-like User-Agent required by NRL.com
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

SCRAPE_DELAY = 2.0  # seconds between NRL.com requests

# ---------------------------------------------------------------------------
# Canonical team name -> NRL.com URL slug
# ---------------------------------------------------------------------------
NRL_URL_SLUGS: dict[str, str] = {
    "Brisbane Broncos": "broncos",
    "North Queensland Cowboys": "cowboys",
    "Gold Coast Titans": "titans",
    "Dolphins": "dolphins",
    "Melbourne Storm": "storm",
    "New Zealand Warriors": "warriors",
    "Penrith Panthers": "panthers",
    "Sydney Roosters": "roosters",
    "South Sydney Rabbitohs": "rabbitohs",
    "Canterbury Bulldogs": "bulldogs",
    "Manly Sea Eagles": "sea-eagles",
    "Newcastle Knights": "knights",
    "Canberra Raiders": "raiders",
    "Wests Tigers": "wests-tigers",
    "Cronulla Sharks": "sharks",
    "Parramatta Eels": "eels",
    "St George Illawarra Dragons": "dragons",
}

# Reverse lookup: slug -> canonical name
_SLUG_TO_CANONICAL = {v: k for k, v in NRL_URL_SLUGS.items()}

# S3 data uses short nicknames; map them to canonical names
_S3_NICKNAME_MAP: dict[str, str] = {
    "Broncos": "Brisbane Broncos",
    "Cowboys": "North Queensland Cowboys",
    "Titans": "Gold Coast Titans",
    "Dolphins": "Dolphins",
    "Storm": "Melbourne Storm",
    "Warriors": "New Zealand Warriors",
    "Panthers": "Penrith Panthers",
    "Roosters": "Sydney Roosters",
    "Rabbitohs": "South Sydney Rabbitohs",
    "Bulldogs": "Canterbury Bulldogs",
    "Sea Eagles": "Manly Sea Eagles",
    "Knights": "Newcastle Knights",
    "Raiders": "Canberra Raiders",
    "Wests Tigers": "Wests Tigers",
    "Tigers": "Wests Tigers",
    "Sharks": "Cronulla Sharks",
    "Eels": "Parramatta Eels",
    "Dragons": "St George Illawarra Dragons",
}

# Stats we want to extract (stat title from NRL.com API -> our column suffix)
DESIRED_STATS: dict[str, str] = {
    "Completion Rate": "completion_rate",
    "All Run Metres": "run_metres",
    "Tackles Made": "tackles",
    "Missed Tackles": "missed_tackles",
    "Errors": "errors",
    "Offloads": "offloads",
    "Line Breaks": "line_breaks",
    "Tackle Breaks": "line_break_assists",  # closest equivalent
    "Kicking Metres": "kick_metres",
    "Possession %": "possession_pct",
    "Penalties Conceded": "penalties_conceded",
    "Post Contact Metres": "post_contact_metres",
    "All Runs": "all_runs",
    "Effective Tackle %": "effective_tackle_pct",
    "Forced Drop Outs": "forced_drop_outs",
    "Time In Possession": "time_in_possession",
    "Average Set Distance": "avg_set_distance",
    "Intercepts": "intercepts",
    "Ineffective Tackles": "ineffective_tackles",
}

# S3 data key -> our column suffix (same stats, different key names)
S3_STAT_KEYS: dict[str, str] = {
    "Completion Rate": "completion_rate",
    "all_run_metres": "run_metres",
    "tackles_made": "tackles",
    "missed_tackles": "missed_tackles",
    "errors": "errors",
    "offloads": "offloads",
    "line_breaks": "line_breaks",
    "tackle_breaks": "line_break_assists",
    "kicking_metres": "kick_metres",
    # possession_pct not directly available from S3 (would need to compute)
    "penalties_conceded": "penalties_conceded",
    "post_contact_metres": "post_contact_metres",
    "all_runs": "all_runs",
    "Effective_Tackle": "effective_tackle_pct",
    "forced_drop_outs": "forced_drop_outs",
    "time_in_possession": "time_in_possession",
    "average_set_distance": "avg_set_distance",
    "intercepts": "intercepts",
    "ineffective_tackles": "ineffective_tackles",
}

# Map our internal round identifiers to NRL.com URL round slugs.
# Regular season rounds: "1" -> "round-1", etc.
# Finals: our data uses "semi-final", "prelim-final", "grand-final".
# NRL.com uses different patterns depending on year:
#   - Modern (2018+): "finals-week-1" through "finals-week-3" + "grand-final"
#   - Older: "round-N" where N is the round number beyond the regular season
# We try all patterns to maximise coverage.
FINALS_URL_SLUGS_NAMED = [
    "finals-week-1",
    "finals-week-2",
    "finals-week-3",
    "grand-final",
]


# ============================================================================
# Helper functions
# ============================================================================

def _get_session() -> requests.Session:
    """Create a requests session with browser-like headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-AU,en;q=0.9",
    })
    return session


def _cache_path_nrl(year: int, round_str: str, home_slug: str,
                    away_slug: str) -> Path:
    """Return the local cache file path for an NRL.com API response."""
    safe_round = round_str.replace(" ", "_")
    filename = f"{year}__{safe_round}__{home_slug}_v_{away_slug}.json"
    return CACHE_DIR / "nrl_api" / filename


def _cache_path_s3(year: int) -> Path:
    """Return the local cache file path for an S3 bulk download."""
    return CACHE_DIR / "s3_bulk" / f"NRL_detailed_match_data_{year}.json"


def _parse_numeric(value: Any) -> float | None:
    """Parse a stat value to float, handling commas and percentages."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    # Remove commas (e.g. "1,701")
    s = s.replace(",", "")
    # Remove percentage sign
    s = s.rstrip("%")
    # Remove 's' suffix for seconds (e.g. "3.82s")
    s = s.rstrip("s")
    # Handle time format like "32:08" -> convert to seconds
    if re.match(r"^\d+:\d+$", s):
        parts = s.split(":")
        return float(parts[0]) * 60 + float(parts[1])
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _get_nrl_url_slug(canonical_name: str) -> str:
    """Get the NRL.com URL slug for a canonical team name."""
    slug = NRL_URL_SLUGS.get(canonical_name)
    if slug is None:
        raise ValueError(f"No NRL.com URL slug for team: {canonical_name}")
    return slug


def _round_to_url_slug(round_str: str) -> str | None:
    """Convert a round string from matches.parquet to an NRL.com URL slug.

    Returns None for finals rounds (handled separately).
    """
    # Regular season rounds
    try:
        round_num = int(round_str)
        return f"round-{round_num}"
    except ValueError:
        pass
    # Finals rounds return None; caller will try multiple slugs
    return None


def _s3_nickname_to_canonical(nickname: str) -> str | None:
    """Resolve a short team nickname from S3 data to canonical name."""
    canonical = _S3_NICKNAME_MAP.get(nickname)
    if canonical:
        return canonical
    # Try standardise_team_name as fallback
    try:
        return standardise_team_name(nickname)
    except KeyError:
        return None


# ============================================================================
# S3 bulk data loading
# ============================================================================

def download_s3_data(year: int, session: requests.Session,
                     force: bool = False) -> dict | None:
    """Download and cache the S3 bulk data for a given year.

    Returns the parsed JSON dict, or None on failure.
    """
    cache_file = _cache_path_s3(year)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not force:
        logger.info("S3 %d: loading from cache %s", year, cache_file.name)
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    url = S3_BULK_URL.format(year=year)
    logger.info("S3 %d: downloading from %s", year, url)
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.error("S3 %d: download failed: %s", year, e)
        return None

    data = r.json()
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info("S3 %d: cached %d KB", year, len(r.content) // 1024)
    return data


def _s3_needs_swap(year: int, round_str: str,
                   s3_home: str, s3_away: str,
                   matches_df: pd.DataFrame) -> bool:
    """Check if S3 home/away is reversed relative to matches.parquet.

    Returns True if the teams need to be swapped to match our data.
    """
    year_round = matches_df[
        (matches_df["year"] == year) & (matches_df["round"] == round_str)
    ]
    # Check if the match exists with the S3 ordering
    exact = year_round[
        (year_round["home_team"] == s3_home)
        & (year_round["away_team"] == s3_away)
    ]
    if not exact.empty:
        return False

    # Check if the match exists with reversed ordering
    reversed_match = year_round[
        (year_round["home_team"] == s3_away)
        & (year_round["away_team"] == s3_home)
    ]
    if not reversed_match.empty:
        return True

    return False


def parse_s3_data(data: dict, year: int,
                  matches_df: pd.DataFrame) -> list[dict]:
    """Parse S3 bulk JSON into a list of stat rows.

    Each row is a dict with keys: year, round, home_team, away_team,
    home_{stat}, away_{stat}.
    """
    rows = []
    nrl_data = data.get("NRL", [])

    for round_entry in nrl_data:
        for round_key, match_list in round_entry.items():
            # round_key is "1", "2", ..., "28" (finals start at 28)
            round_num = int(round_key)

            for match_dict in match_list:
                for match_name, match_data in match_dict.items():
                    # match_name is like "Eels v Storm"
                    parts = match_name.split(" v ")
                    if len(parts) != 2:
                        logger.warning(
                            "S3 %d R%s: unexpected match name '%s'",
                            year, round_key, match_name,
                        )
                        continue

                    home_nick = parts[0].strip()
                    away_nick = parts[1].strip()
                    home_team = _s3_nickname_to_canonical(home_nick)
                    away_team = _s3_nickname_to_canonical(away_nick)

                    if not home_team or not away_team:
                        logger.warning(
                            "S3 %d R%s: cannot resolve teams '%s' v '%s'",
                            year, round_key, home_nick, away_nick,
                        )
                        continue

                    home_stats_raw = match_data.get("home", {})
                    away_stats_raw = match_data.get("away", {})

                    # Map S3 round number to our round identifier
                    # Regular rounds 1-27 map directly; 28+ are finals
                    actual_round = _map_s3_round_to_ours(
                        round_num, year, home_team, away_team, matches_df,
                    )

                    # Check if S3 home/away is reversed relative to
                    # matches.parquet. If so, swap teams and stats.
                    needs_swap = _s3_needs_swap(
                        year, actual_round, home_team, away_team, matches_df,
                    )
                    if needs_swap:
                        home_team, away_team = away_team, home_team
                        home_stats_raw, away_stats_raw = (
                            away_stats_raw, home_stats_raw,
                        )

                    row: dict[str, Any] = {
                        "year": year,
                        "round": actual_round,
                        "home_team": home_team,
                        "away_team": away_team,
                    }

                    # Extract stats from home and away
                    for s3_key, col_suffix in S3_STAT_KEYS.items():
                        h_val = home_stats_raw.get(s3_key)
                        a_val = away_stats_raw.get(s3_key)
                        row[f"home_{col_suffix}"] = _parse_numeric(h_val)
                        row[f"away_{col_suffix}"] = _parse_numeric(a_val)

                    rows.append(row)

    return rows


def _get_max_regular_round(year: int, matches_df: pd.DataFrame) -> int:
    """Get the highest numbered regular-season round for a given year."""
    year_matches = matches_df[matches_df["year"] == year]
    numeric_rounds = [
        int(r) for r in year_matches["round"].unique() if r.isdigit()
    ]
    return max(numeric_rounds) if numeric_rounds else 27


def _map_s3_round_to_ours(s3_round: int, year: int,
                          home_team: str, away_team: str,
                          matches_df: pd.DataFrame) -> str:
    """Map S3 round numbers to our round identifiers.

    Regular season rounds map directly. Rounds beyond the max regular
    round for the year are treated as finals and mapped by matching
    teams against matches.parquet finals rows.
    """
    max_regular = _get_max_regular_round(year, matches_df)

    if s3_round <= max_regular:
        return str(s3_round)

    # For finals, search ONLY finals rounds in matches_df to avoid
    # matching a regular-season rematch of the same two teams.
    finals_rounds = {"semi-final", "prelim-final", "grand-final",
                     "qualif-final", "elim-final"}
    year_finals = matches_df[
        (matches_df["year"] == year)
        & (matches_df["round"].isin(finals_rounds))
    ]
    for _, row in year_finals.iterrows():
        if ((row["home_team"] == home_team and row["away_team"] == away_team)
                or (row["home_team"] == away_team
                    and row["away_team"] == home_team)):
            return str(row["round"])

    # Fallback: use a rough positional mapping based on NRL finals structure.
    # The offset from max_regular tells us which finals week:
    # +1/+2/+3/+4 = week 1 (QF/EF), +5/+6 = semi, +7/+8 = prelim, etc.
    offset = s3_round - max_regular
    if offset <= 4:
        return "semi-final"   # Week 1 (QF/EF) -> closest equivalent
    elif offset <= 6:
        return "semi-final"
    elif offset <= 8:
        return "prelim-final"
    else:
        return "grand-final"


# ============================================================================
# NRL.com API scraping
# ============================================================================

_last_request_time: float = 0.0


def fetch_nrl_api(year: int, round_slug: str, home_slug: str,
                  away_slug: str, session: requests.Session,
                  force: bool = False) -> dict | None:
    """Fetch match data from NRL.com API, with caching and rate limiting.

    Rate limiting (2s delay) is applied automatically before each
    network request. Cached responses are served immediately.

    Returns parsed JSON dict or None on failure.
    """
    global _last_request_time

    cache_file = _cache_path_nrl(year, round_slug, home_slug, away_slug)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not force:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    url = NRL_API_URL.format(
        year=year, round_slug=round_slug,
        home_slug=home_slug, away_slug=away_slug,
    )

    # Rate limit: wait at least SCRAPE_DELAY since last request
    elapsed = time.time() - _last_request_time
    if elapsed < SCRAPE_DELAY:
        time.sleep(SCRAPE_DELAY - elapsed)

    try:
        _last_request_time = time.time()
        r = session.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
    except requests.RequestException as e:
        logger.debug("NRL API %s: %s", url, e)
        return None

    data = r.json()
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def parse_nrl_api_stats(data: dict) -> dict[str, float | None]:
    """Extract team-level stats from NRL.com API response.

    Returns a dict like:
        {"home_completion_rate": 80.0, "away_completion_rate": 79.0, ...}
    """
    result: dict[str, float | None] = {}
    stats = data.get("stats", {})
    groups = stats.get("groups", [])

    for group in groups:
        for stat in group.get("stats", []):
            title = stat.get("title", "")
            col_suffix = DESIRED_STATS.get(title)
            if col_suffix is None:
                continue

            home_val = stat.get("homeValue", {})
            away_val = stat.get("awayValue", {})
            result[f"home_{col_suffix}"] = (
                _parse_numeric(home_val.get("value"))
            )
            result[f"away_{col_suffix}"] = (
                _parse_numeric(away_val.get("value"))
            )

    return result


def _get_finals_slugs_to_try(round_str: str,
                             max_regular_round: int) -> list[str]:
    """Return an ordered list of URL round slugs to try for a finals match.

    The order is critical: the same two teams can meet in multiple
    finals weeks (e.g. qualifying final + grand final), so we must
    try the most specific slug first.

    NRL finals structure:
        Week 1: Qualifying Finals (1v4, 2v3) + Elimination Finals (5v8, 6v7)
        Week 2: Semi-Finals (loser QF1 v winner EF1, loser QF2 v winner EF2)
        Week 3: Preliminary Finals (winner QF1 v winner SF1, etc.)
        Grand Final

    URL formats:
        Modern (varies by year): finals-week-1, finals-week-2, finals-week-3, grand-final
        Older: round-{max_regular+1}, round-{max_regular+2}, etc.
    """
    mr = max_regular_round
    if round_str == "grand-final":
        return [
            "grand-final",
            f"round-{mr + 4}",  # GF is 4 rounds after last regular
            f"round-{mr + 5}",  # sometimes offset varies
            f"round-{mr + 3}",
        ]
    elif round_str == "prelim-final":
        return [
            "finals-week-3",
            f"round-{mr + 3}",
            f"round-{mr + 2}",
            f"round-{mr + 4}",
        ]
    elif round_str == "semi-final":
        return [
            "finals-week-2",
            f"round-{mr + 2}",
            "finals-week-1",
            f"round-{mr + 1}",
            f"round-{mr + 3}",
        ]
    elif round_str in ("qualif-final", "elim-final"):
        return [
            "finals-week-1",
            f"round-{mr + 1}",
            f"round-{mr + 2}",
        ]
    else:
        # Generic fallback: try everything
        return [
            "finals-week-1",
            "finals-week-2",
            "finals-week-3",
            "grand-final",
            f"round-{mr + 1}",
            f"round-{mr + 2}",
            f"round-{mr + 3}",
            f"round-{mr + 4}",
        ]


def scrape_match_nrl_api(
    year: int,
    round_str: str,
    home_team: str,
    away_team: str,
    session: requests.Session,
    force: bool = False,
    max_regular_round: int = 27,
) -> dict | None:
    """Scrape a single match from NRL.com API.

    Tries home-v-away URL first, then away-v-home if 404.
    For finals rounds, tries multiple round slug patterns.

    Parameters
    ----------
    max_regular_round : int
        The highest regular-season round number for this year.
        Used to compute round-N URL slugs for older finals format.

    Returns a stat row dict or None if data cannot be fetched.
    """
    home_slug = _get_nrl_url_slug(home_team)
    away_slug = _get_nrl_url_slug(away_team)

    # Determine round slugs to try
    url_round_slug = _round_to_url_slug(round_str)
    if url_round_slug is not None:
        round_slugs_to_try = [url_round_slug]
    else:
        # Finals match: try named slugs (modern format) first, then
        # round-N patterns (older format). Order matters because the
        # same teams can play in multiple finals weeks, so we must
        # try the slug most likely to match the specific round first.
        round_slugs_to_try = _get_finals_slugs_to_try(
            round_str, max_regular_round,
        )

    for round_slug in round_slugs_to_try:
        # Try home-v-away first
        data = fetch_nrl_api(year, round_slug, home_slug, away_slug,
                             session, force)
        if data is not None:
            stats = parse_nrl_api_stats(data)
            # Verify we got the right teams
            api_home = data.get("homeTeam", {}).get("name", "")
            api_away = data.get("awayTeam", {}).get("name", "")
            stats["_api_home"] = api_home
            stats["_api_away"] = api_away
            stats["_url_round_slug"] = round_slug
            stats["_swapped"] = False
            return stats

        # Try reversed order
        data = fetch_nrl_api(year, round_slug, away_slug, home_slug,
                             session, force)
        if data is not None:
            stats = parse_nrl_api_stats(data)
            api_home = data.get("homeTeam", {}).get("name", "")
            api_away = data.get("awayTeam", {}).get("name", "")

            # The API returns data relative to its own home/away.
            # If the URL order was reversed, we need to check if
            # the API's home matches our home.
            try:
                api_home_canonical = standardise_team_name(api_home)
            except KeyError:
                api_home_canonical = api_home
            try:
                api_away_canonical = standardise_team_name(api_away)
            except KeyError:
                api_away_canonical = api_away

            if api_home_canonical == away_team:
                # API's home is our away team -> swap home/away stats
                swapped: dict[str, Any] = {}
                for key, val in stats.items():
                    if key.startswith("home_"):
                        swapped[key.replace("home_", "away_", 1)] = val
                    elif key.startswith("away_"):
                        swapped[key.replace("away_", "home_", 1)] = val
                    else:
                        swapped[key] = val
                swapped["_api_home"] = api_home
                swapped["_api_away"] = api_away
                swapped["_url_round_slug"] = round_slug
                swapped["_swapped"] = True
                return swapped
            else:
                stats["_api_home"] = api_home
                stats["_api_away"] = api_away
                stats["_url_round_slug"] = round_slug
                stats["_swapped"] = False
                return stats

    return None


# ============================================================================
# Main pipeline
# ============================================================================

def load_matches() -> pd.DataFrame:
    """Load the existing matches.parquet."""
    path = PROCESSED_DIR / "matches.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"matches.parquet not found at {path}. "
            f"Run run_scrape.py first to generate it."
        )
    df = pd.read_parquet(path)
    # Ensure round is string
    df["round"] = df["round"].astype(str)
    return df


def build_match_stats(
    years: list[int] | None = None,
    force_rescrape: bool = False,
) -> pd.DataFrame:
    """Main function: scrape/load match stats and return a DataFrame.

    Parameters
    ----------
    years : list of ints, optional
        Restrict to these years. If None, process all years in matches.parquet.
    force_rescrape : bool
        If True, ignore cached data and re-download everything.

    Returns
    -------
    pd.DataFrame
        Match stats with columns: year, round, home_team, away_team,
        home_{stat}, away_{stat} for each stat.
    """
    matches_df = load_matches()

    # Filter to non-bye, non-walkover, non-abandoned matches
    mask = ~matches_df["is_bye"] & ~matches_df["is_walkover"]
    if "is_abandoned" in matches_df.columns:
        mask = mask & ~matches_df["is_abandoned"]
    matches_df = matches_df[mask].copy()

    if years is not None:
        matches_df = matches_df[matches_df["year"].isin(years)].copy()

    all_years = sorted(matches_df["year"].unique())
    logger.info("Processing %d years: %s", len(all_years), all_years)

    session = _get_session()
    all_rows: list[dict] = []

    # -------------------------------------------------------------------
    # Phase 1: Load S3 bulk data for 2021-2024
    # -------------------------------------------------------------------
    s3_covered_years = set()
    for year in all_years:
        if year in S3_YEARS:
            logger.info("=" * 60)
            logger.info("Phase 1 - S3 bulk data for %d", year)
            data = download_s3_data(year, session, force=force_rescrape)
            if data is not None:
                rows = parse_s3_data(data, year, matches_df)
                logger.info(
                    "S3 %d: parsed %d match stat rows", year, len(rows),
                )
                all_rows.extend(rows)
                s3_covered_years.add(year)
            else:
                logger.warning(
                    "S3 %d: failed to load, will fall back to NRL.com API",
                    year,
                )

    # -------------------------------------------------------------------
    # Phase 2: Scrape NRL.com API for remaining years
    # -------------------------------------------------------------------
    api_years = [y for y in all_years if y not in s3_covered_years]
    if api_years:
        logger.info("=" * 60)
        logger.info(
            "Phase 2 - NRL.com API for years: %s", api_years,
        )

    for year in api_years:
        year_matches = matches_df[matches_df["year"] == year]
        max_regular = _get_max_regular_round(year, matches_df)
        logger.info("-" * 50)
        logger.info(
            "NRL.com API: %d (%d matches, max_regular_round=%d)",
            year, len(year_matches), max_regular,
        )

        success = 0
        fail = 0
        for idx, (_, match) in enumerate(year_matches.iterrows()):
            home_team = match["home_team"]
            away_team = match["away_team"]
            round_str = str(match["round"])

            logger.debug(
                "  [%d/%d] %s R%s: %s v %s",
                idx + 1, len(year_matches), year, round_str,
                home_team, away_team,
            )

            stats = scrape_match_nrl_api(
                year, round_str, home_team, away_team,
                session, force=force_rescrape,
                max_regular_round=max_regular,
            )

            if stats is not None:
                # Remove internal metadata keys
                row = {
                    "year": year,
                    "round": round_str,
                    "home_team": home_team,
                    "away_team": away_team,
                }
                for key, val in stats.items():
                    if not key.startswith("_"):
                        row[key] = val
                all_rows.append(row)
                success += 1
            else:
                fail += 1
                logger.warning(
                    "  MISS: %d R%s %s v %s",
                    year, round_str, home_team, away_team,
                )

            # Rate limiting is handled inside fetch_nrl_api

            # Progress logging every 20 matches
            if (idx + 1) % 20 == 0:
                logger.info(
                    "  Progress: %d/%d (success=%d, miss=%d)",
                    idx + 1, len(year_matches), success, fail,
                )

        logger.info(
            "NRL.com %d: %d/%d matches scraped successfully",
            year, success, success + fail,
        )

        # Save progress incrementally after each year
        _save_progress(all_rows)

    # -------------------------------------------------------------------
    # Phase 3: Build and save final DataFrame
    # -------------------------------------------------------------------
    if not all_rows:
        logger.warning("No match stats collected!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Ensure consistent column ordering
    id_cols = ["year", "round", "home_team", "away_team"]
    stat_cols = sorted([c for c in df.columns if c not in id_cols])
    df = df[id_cols + stat_cols]

    # Drop duplicates (S3 and API might overlap)
    df = df.drop_duplicates(
        subset=["year", "round", "home_team", "away_team"],
        keep="first",
    )

    # Filter to only keep rows that match entries in matches.parquet.
    # S3 data may include extra matches (qualifying/elimination finals)
    # that our dataset doesn't track. Also handles cases where S3 has
    # home/away reversed relative to our data.
    match_keys = set(
        zip(
            matches_df["year"],
            matches_df["round"],
            matches_df["home_team"],
            matches_df["away_team"],
        )
    )
    mask = df.apply(
        lambda r: (r["year"], r["round"], r["home_team"], r["away_team"])
        in match_keys,
        axis=1,
    )
    n_before = len(df)
    df = df[mask].copy()
    if len(df) < n_before:
        logger.info(
            "Filtered %d -> %d rows (removed %d unmatched entries)",
            n_before, len(df), n_before - len(df),
        )

    # Sort
    df = df.sort_values(["year", "round", "home_team"]).reset_index(drop=True)

    # Add parsed_date from matches_df for convenience
    date_lookup = matches_df.set_index(
        ["year", "round", "home_team", "away_team"]
    )["parsed_date"].to_dict()

    df["date"] = df.apply(
        lambda r: date_lookup.get(
            (r["year"], r["round"], r["home_team"], r["away_team"])
        ),
        axis=1,
    )

    # Reorder with date after away_team
    id_cols_with_date = ["year", "round", "date", "home_team", "away_team"]
    stat_cols = [c for c in df.columns if c not in id_cols_with_date]
    df = df[id_cols_with_date + stat_cols]

    return df


def _save_progress(rows: list[dict]) -> None:
    """Save intermediate results to a temporary parquet file."""
    if not rows:
        return
    tmp_path = PROCESSED_DIR / "match_stats_partial.parquet"
    try:
        pd.DataFrame(rows).to_parquet(tmp_path, index=False, engine="pyarrow")
    except Exception as e:
        logger.warning("Could not save progress: %s", e)


def save_match_stats(df: pd.DataFrame) -> Path:
    """Save the final match_stats.parquet."""
    output_path = PROCESSED_DIR / "match_stats.parquet"
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Saved %d rows to %s", len(df), output_path)

    # Clean up partial file if it exists
    partial = PROCESSED_DIR / "match_stats_partial.parquet"
    if partial.exists():
        partial.unlink()
        logger.info("Removed partial progress file")

    return output_path


# ============================================================================
# Rolling feature computation
# ============================================================================

def load_match_stats() -> pd.DataFrame:
    """Load the saved match_stats.parquet."""
    path = PROCESSED_DIR / "match_stats.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"match_stats.parquet not found at {path}. "
            f"Run this script first to generate it."
        )
    return pd.read_parquet(path)


def compute_rolling_match_stats(
    df: pd.DataFrame | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling averages of match stats per team.

    For each match, computes rolling averages of key stats over the
    previous N games for both home and away teams. These are *pre-match*
    features (the current match is excluded from the rolling window).

    Parameters
    ----------
    df : pd.DataFrame, optional
        Match stats DataFrame. If None, loads from match_stats.parquet.
    windows : list of int, optional
        Rolling window sizes in number of games. Defaults to [3, 5, 8].

    Returns
    -------
    pd.DataFrame
        One row per match with columns:
        year, round, date, home_team, away_team,
        home_rolling_{stat}_{window}, away_rolling_{stat}_{window}
    """
    if df is None:
        df = load_match_stats()

    if windows is None:
        windows = [3, 5, 8]

    # Stats to compute rolling averages for
    rolling_stats = [
        "completion_rate", "run_metres", "tackles", "missed_tackles",
        "errors", "offloads", "line_breaks", "line_break_assists",
        "kick_metres", "possession_pct", "penalties_conceded",
        "post_contact_metres", "effective_tackle_pct",
    ]

    # Ensure date is datetime and sorted
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "year", "round"]).reset_index(drop=True)

    # Build a per-team timeline: for each match, record each team's stats
    # We create a "long" format: one row per team per match
    team_records: list[dict] = []
    for _, row in df.iterrows():
        base = {"year": row["year"], "round": row["round"],
                "date": row["date"]}
        # Home team record
        home_rec = {**base, "team": row["home_team"],
                    "opponent": row["away_team"], "is_home": True}
        for stat in rolling_stats:
            col = f"home_{stat}"
            home_rec[stat] = row.get(col)
        team_records.append(home_rec)

        # Away team record
        away_rec = {**base, "team": row["away_team"],
                    "opponent": row["home_team"], "is_home": False}
        for stat in rolling_stats:
            col = f"away_{stat}"
            away_rec[stat] = row.get(col)
        team_records.append(away_rec)

    team_df = pd.DataFrame(team_records)
    team_df = team_df.sort_values(["team", "date"]).reset_index(drop=True)

    # Compute rolling averages per team (shifted by 1 to exclude current match)
    for window in windows:
        for stat in rolling_stats:
            col_name = f"rolling_{stat}_{window}"
            team_df[col_name] = (
                team_df.groupby("team")[stat]
                .transform(lambda x: x.shift(1).rolling(
                    window, min_periods=1,
                ).mean())
            )

    # Merge back: for each match, attach home team's rolling stats and
    # away team's rolling stats
    result = df[["year", "round", "date", "home_team", "away_team"]].copy()

    for window in windows:
        for stat in rolling_stats:
            col_name = f"rolling_{stat}_{window}"

            # Home team's rolling average
            home_lookup = team_df[team_df["is_home"]][
                ["year", "round", "team", col_name]
            ].rename(columns={
                "team": "home_team",
                col_name: f"home_{col_name}",
            })
            result = result.merge(
                home_lookup,
                on=["year", "round", "home_team"],
                how="left",
            )

            # Away team's rolling average
            away_lookup = team_df[~team_df["is_home"]][
                ["year", "round", "team", col_name]
            ].rename(columns={
                "team": "away_team",
                col_name: f"away_{col_name}",
            })
            result = result.merge(
                away_lookup,
                on=["year", "round", "away_team"],
                how="left",
            )

    # Add differentials for each rolling stat
    for window in windows:
        for stat in rolling_stats:
            home_col = f"home_rolling_{stat}_{window}"
            away_col = f"away_rolling_{stat}_{window}"
            diff_col = f"diff_rolling_{stat}_{window}"
            if home_col in result.columns and away_col in result.columns:
                result[diff_col] = result[home_col] - result[away_col]

    return result


# ============================================================================
# CLI entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape NRL match statistics from NRL.com API and S3.",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=None,
        help="Specific years to scrape (default: all 2013-2025)",
    )
    parser.add_argument(
        "--force-rescrape", action="store_true",
        help="Ignore cache and re-download all data",
    )
    parser.add_argument(
        "--rolling-only", action="store_true",
        help="Skip scraping; just compute rolling features from existing data",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    t_start = time.time()

    print("=" * 70)
    print("  NRL Match Stats Scraper")
    print("=" * 70)

    if args.rolling_only:
        print("\n  Computing rolling features from existing match_stats...")
        rolling_df = compute_rolling_match_stats()
        rolling_path = PROCESSED_DIR / "match_stats_rolling.parquet"
        rolling_df.to_parquet(rolling_path, index=False, engine="pyarrow")
        print(f"  Saved {len(rolling_df)} rows to {rolling_path}")
        print(f"  Columns: {len(rolling_df.columns)}")
        print(f"  Rolling stat columns: "
              f"{len([c for c in rolling_df.columns if 'rolling' in c])}")
    else:
        # Run scraping
        df = build_match_stats(
            years=args.years,
            force_rescrape=args.force_rescrape,
        )

        if df.empty:
            print("\n  No data collected. Check logs for errors.")
            return

        output_path = save_match_stats(df)

        # Summary
        print(f"\n{'=' * 70}")
        print("  Summary")
        print(f"{'=' * 70}")
        print(f"  Total match stat rows: {len(df)}")
        print(f"  Years: {sorted(df['year'].unique())}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Output: {output_path}")

        # Per-year counts
        print("\n  Matches per year:")
        for year, count in df.groupby("year").size().items():
            print(f"    {year}: {count}")

        # Stat coverage
        stat_cols = [c for c in df.columns
                     if c.startswith("home_") or c.startswith("away_")]
        print(f"\n  Stat columns ({len(stat_cols)}):")
        for col in sorted(stat_cols):
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"    {col}: {non_null}/{len(df)} ({pct:.0f}%)")

        # Sample
        print(f"\n  Sample (first 3 rows):")
        print(df.head(3).to_string(index=False))

        # Also compute rolling features
        print(f"\n{'=' * 70}")
        print("  Computing rolling features...")
        print(f"{'=' * 70}")
        rolling_df = compute_rolling_match_stats(df)
        rolling_path = PROCESSED_DIR / "match_stats_rolling.parquet"
        rolling_df.to_parquet(rolling_path, index=False, engine="pyarrow")
        print(f"  Saved {len(rolling_df)} rows with "
              f"{len(rolling_df.columns)} columns to {rolling_path}")

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
