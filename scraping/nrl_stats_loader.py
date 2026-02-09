"""
Load NRL match statistics from the beauhobba/NRL-Data S3 JSON files.

The `beauhobba/NRL-Data <https://github.com/beauhobba/NRL-Data>`_ project
scrapes the official NRL website and publishes JSON files to a public S3
bucket.  These files contain detailed per-match and per-player statistics
that are not available on RLP:

- **Match-level**: possession %, completion rate, run metres, tackles,
  errors, offloads, line breaks, kick metres, etc.
- **Player-level**: tackles, runs, run metres, try assists, line break
  assists, etc.

This module provides functions to:

1. Download (and cache) raw JSON files from S3.
2. Parse match-level stats into a flat DataFrame.
3. Parse player-level stats into a long-form DataFrame.
4. Standardise team names to match the rest of the pipeline.

Public API
----------
- :func:`load_match_stats`  -- load and parse match-level stats.
- :func:`load_player_stats` -- load and parse player-level stats.
- :func:`fetch_nrl_data_json` -- download a single JSON from S3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from config.settings import RAW_DIR, USER_AGENT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S3 endpoint configuration
# ---------------------------------------------------------------------------
# The beauhobba/NRL-Data scraper publishes to this public S3 bucket.
# Files are organised as:
#   s3://nrl-data-public/{year}/{round}/{match_id}.json
#
# Since the exact bucket URL may change, allow override via an env var.
import os

NRL_DATA_BASE_URL: str = os.getenv(
    "NRL_DATA_BASE_URL",
    "https://nrl-data-public.s3.ap-southeast-2.amazonaws.com",
)

# Local cache directory for downloaded JSON files.
NRL_STATS_DIR: Path = RAW_DIR / "nrl_stats"
NRL_STATS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Team-name standardisation (reuses odds_loader helper)
# ---------------------------------------------------------------------------

def _standardise(name: str) -> str:
    """Standardise a team name via the project-wide alias map."""
    try:
        from scraping.odds_loader import standardise_team_name
        return standardise_team_name(name)
    except ImportError:
        return name.strip()


# ---------------------------------------------------------------------------
# JSON fetching / caching
# ---------------------------------------------------------------------------

def fetch_nrl_data_json(
    url: str,
    *,
    cache_dir: Path = NRL_STATS_DIR,
    force: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Download a JSON file from the NRL-Data S3 bucket and return its
    parsed content.

    Parameters
    ----------
    url:
        Fully-qualified URL to the JSON resource.
    cache_dir:
        Local directory for caching downloaded files.
    force:
        If ``True``, ignore any cached copy and re-download.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    dict
        Parsed JSON content.

    Raises
    ------
    requests.HTTPError
        On non-2xx HTTP responses.
    json.JSONDecodeError
        If the response body is not valid JSON.
    """
    # Derive a cache filename from the URL path.
    from urllib.parse import urlparse

    parsed = urlparse(url)
    rel_path = parsed.path.lstrip("/")
    cache_file = cache_dir / rel_path

    if cache_file.is_file() and not force:
        logger.debug("NRL-Data cache hit: %s", cache_file)
        return json.loads(cache_file.read_text(encoding="utf-8"))

    logger.info("Downloading NRL-Data JSON: %s", url)
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    # Cache locally.
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return data


def _load_local_json(path: Path) -> dict[str, Any]:
    """Load a single JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Match-level stats parser
# ---------------------------------------------------------------------------

# Standard match-stat keys expected in the JSON.  Keys are normalised to
# lowercase-underscore form.
_MATCH_STAT_KEYS: list[str] = [
    "possession_pct",
    "completion_rate",
    "run_metres",
    "tackles",
    "missed_tackles",
    "errors",
    "offloads",
    "line_breaks",
    "line_break_assists",
    "kick_metres",
    "kicks",
    "penalties_conceded",
    "set_restarts",
    "interchanges_used",
]


def _normalise_key(key: str) -> str:
    """Normalise a JSON key to snake_case."""
    import re
    # CamelCase -> snake_case
    s = re.sub(r"([A-Z])", r"_\1", key).lower().strip("_")
    # Replace whitespace / hyphens with underscores.
    s = re.sub(r"[\s\-]+", "_", s)
    # Collapse multiple underscores.
    s = re.sub(r"_+", "_", s)
    return s


def parse_match_stats_json(data: dict[str, Any]) -> dict[str, Any]:
    """Parse a single match JSON object into a flat stats dict.

    The returned dict contains:

    - ``match_id``, ``year``, ``round``, ``date``
    - ``home_team``, ``away_team``
    - ``home_score``, ``away_score``
    - For each stat in :data:`_MATCH_STAT_KEYS`:
      ``home_{stat}`` and ``away_{stat}``.

    Parameters
    ----------
    data:
        Raw parsed JSON for a single match.

    Returns
    -------
    dict
        Flat dictionary of match-level statistics.
    """
    result: dict[str, Any] = {}

    # Metadata.
    result["match_id"] = data.get("matchId") or data.get("match_id")
    result["year"] = data.get("year") or data.get("season")
    result["round"] = data.get("round") or data.get("roundNumber")
    result["date"] = data.get("date") or data.get("startTime")

    # Teams.
    home_data = data.get("homeTeam") or data.get("home") or {}
    away_data = data.get("awayTeam") or data.get("away") or {}

    if isinstance(home_data, str):
        result["home_team"] = _standardise(home_data)
        result["away_team"] = _standardise(str(away_data))
    else:
        result["home_team"] = _standardise(
            home_data.get("name") or home_data.get("teamName") or ""
        )
        result["away_team"] = _standardise(
            away_data.get("name") or away_data.get("teamName") or ""
        )

    # Scores.
    result["home_score"] = (
        data.get("homeScore")
        or (home_data.get("score") if isinstance(home_data, dict) else None)
    )
    result["away_score"] = (
        data.get("awayScore")
        or (away_data.get("score") if isinstance(away_data, dict) else None)
    )

    # Stats -- try nested "stats" dict per team, or top-level dict.
    home_stats = (
        home_data.get("stats", {}) if isinstance(home_data, dict) else {}
    )
    away_stats = (
        away_data.get("stats", {}) if isinstance(away_data, dict) else {}
    )

    # Also check for top-level "matchStats" or "teamStats".
    if not home_stats:
        match_stats = data.get("matchStats") or data.get("teamStats") or {}
        if isinstance(match_stats, dict):
            home_stats = match_stats.get("home") or match_stats.get("homeTeam") or {}
            away_stats = match_stats.get("away") or match_stats.get("awayTeam") or {}

    # Normalise all stat keys and extract values.
    for raw_key, value in home_stats.items():
        norm = _normalise_key(raw_key)
        result[f"home_{norm}"] = _to_numeric(value)

    for raw_key, value in away_stats.items():
        norm = _normalise_key(raw_key)
        result[f"away_{norm}"] = _to_numeric(value)

    return result


def _to_numeric(value: Any) -> Optional[float]:
    """Attempt to convert *value* to a float.  Return ``None`` on failure."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Handle percentage strings like "52.3%".
        cleaned = value.strip().rstrip("%")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Player-level stats parser
# ---------------------------------------------------------------------------

def parse_player_stats_json(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse player-level stats from a match JSON into a list of dicts.

    Each dict represents one player's stats for this match with keys:

    - ``match_id``, ``year``, ``round``
    - ``team``, ``player_name``, ``player_id``
    - ``position``
    - Individual stat columns: ``tackles``, ``runs``, ``run_metres``,
      ``try_assists``, ``line_break_assists``, etc.

    Parameters
    ----------
    data:
        Raw parsed JSON for a single match.

    Returns
    -------
    list[dict]
        One dict per player who appeared in the match.
    """
    match_id = data.get("matchId") or data.get("match_id")
    year = data.get("year") or data.get("season")
    round_num = data.get("round") or data.get("roundNumber")

    rows: list[dict[str, Any]] = []

    for side_key, team_key in [("homeTeam", "home"), ("awayTeam", "away")]:
        team_data = data.get(side_key) or data.get(team_key) or {}
        if isinstance(team_data, str):
            team_name = _standardise(team_data)
            players_data: list[dict] = []
        else:
            team_name = _standardise(
                team_data.get("name") or team_data.get("teamName") or ""
            )
            players_data = (
                team_data.get("players")
                or team_data.get("playerStats")
                or []
            )

        # Also look for top-level "playerStats".
        if not players_data:
            top_players = data.get("playerStats") or {}
            if isinstance(top_players, dict):
                players_data = top_players.get(team_key) or top_players.get(side_key) or []

        for player in players_data:
            if not isinstance(player, dict):
                continue

            row: dict[str, Any] = {
                "match_id": match_id,
                "year": year,
                "round": round_num,
                "team": team_name,
                "player_name": (
                    player.get("name")
                    or player.get("playerName")
                    or player.get("fullName")
                    or ""
                ).strip(),
                "player_id": player.get("playerId") or player.get("id"),
                "position": player.get("position") or player.get("positionName"),
            }

            # Extract all numeric stat fields.
            stats = player.get("stats") or player
            for key, value in stats.items():
                if key in ("name", "playerName", "fullName", "playerId",
                           "id", "position", "positionName", "stats"):
                    continue
                norm = _normalise_key(key)
                row[norm] = _to_numeric(value)

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------

def load_match_stats(
    source: str | Path | None = None,
    *,
    year: Optional[int] = None,
    standardise_teams: bool = True,
) -> pd.DataFrame:
    """Load match-level stats into a DataFrame.

    Parameters
    ----------
    source:
        Either a directory containing JSON files, a single JSON file path,
        or ``None`` to use the default cache directory
        (``data/raw/nrl_stats/``).
    year:
        If provided, only load files for this season.
    standardise_teams:
        Whether to apply team-name standardisation.

    Returns
    -------
    pd.DataFrame
        One row per match with all available stats as columns.
    """
    json_files = _resolve_json_files(source, year)

    if not json_files:
        logger.warning("No NRL-Data JSON files found at %s", source or NRL_STATS_DIR)
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for fp in json_files:
        try:
            data = _load_local_json(fp)
            # Some files contain a list of matches; others a single match.
            if isinstance(data, list):
                for item in data:
                    records.append(parse_match_stats_json(item))
            else:
                records.append(parse_match_stats_json(data))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", fp, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if standardise_teams:
        for col in ("home_team", "away_team"):
            if col in df.columns:
                df[col] = df[col].astype(str).apply(_standardise)

    logger.info("Loaded %d match stat records.", len(df))
    return df


def load_player_stats(
    source: str | Path | None = None,
    *,
    year: Optional[int] = None,
    standardise_teams: bool = True,
) -> pd.DataFrame:
    """Load player-level stats into a long-form DataFrame.

    Parameters
    ----------
    source:
        Directory of JSON files, single file, or ``None`` for default.
    year:
        Optional year filter.
    standardise_teams:
        Whether to apply team-name standardisation.

    Returns
    -------
    pd.DataFrame
        One row per player per match with all available stats as columns.
    """
    json_files = _resolve_json_files(source, year)

    if not json_files:
        logger.warning("No NRL-Data JSON files found at %s", source or NRL_STATS_DIR)
        return pd.DataFrame()

    all_rows: list[dict[str, Any]] = []
    for fp in json_files:
        try:
            data = _load_local_json(fp)
            if isinstance(data, list):
                for item in data:
                    all_rows.extend(parse_player_stats_json(item))
            else:
                all_rows.extend(parse_player_stats_json(data))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", fp, exc)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    if standardise_teams and "team" in df.columns:
        df["team"] = df["team"].astype(str).apply(_standardise)

    logger.info("Loaded %d player stat records.", len(df))
    return df


# ---------------------------------------------------------------------------
# File resolution helper
# ---------------------------------------------------------------------------

def _resolve_json_files(
    source: str | Path | None,
    year: Optional[int],
) -> list[Path]:
    """Resolve the source argument into a list of JSON file paths."""
    base = Path(source) if source is not None else NRL_STATS_DIR

    if base.is_file():
        return [base]

    if not base.is_dir():
        return []

    # If a year is specified, look for a year subdirectory first.
    if year is not None:
        year_dir = base / str(year)
        if year_dir.is_dir():
            base = year_dir

    return sorted(base.rglob("*.json"))
