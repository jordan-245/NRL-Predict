"""
Extract comprehensive data from cached NRL.com JSON match files.

Parses all 1,754 cached NRL.com API JSON files in data/raw/nrl_match_stats/nrl_api/
and extracts match metadata, player-level stats, enhanced team stats, and computes
rolling player-quality features per team.

Outputs:
  - data/processed/match_metadata.parquet       (one row per match)
  - data/processed/player_match_stats.parquet    (one row per player per match)
  - data/processed/match_stats_enhanced.parquet  (one row per match, all team stats)
  - data/processed/player_quality_features.parquet (rolling player-quality features)

Usage:
    python extract_nrl_json_data.py
    python extract_nrl_json_data.py --skip-rolling
    python extract_nrl_json_data.py --rolling-windows 3 5 8
    python extract_nrl_json_data.py -v
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

import numpy as np
import pandas as pd

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
logger = logging.getLogger("extract_nrl_json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NRL_API_DIR = RAW_DIR / "nrl_match_stats" / "nrl_api"

# NRL.com URL slug -> canonical team name
SLUG_TO_CANONICAL: dict[str, str] = {
    "broncos": "Brisbane Broncos",
    "cowboys": "North Queensland Cowboys",
    "titans": "Gold Coast Titans",
    "dolphins": "Dolphins",
    "storm": "Melbourne Storm",
    "warriors": "New Zealand Warriors",
    "panthers": "Penrith Panthers",
    "roosters": "Sydney Roosters",
    "rabbitohs": "South Sydney Rabbitohs",
    "bulldogs": "Canterbury Bulldogs",
    "sea-eagles": "Manly Sea Eagles",
    "knights": "Newcastle Knights",
    "raiders": "Canberra Raiders",
    "wests-tigers": "Wests Tigers",
    "sharks": "Cronulla Sharks",
    "eels": "Parramatta Eels",
    "dragons": "St George Illawarra Dragons",
}

# Filename pattern: {year}__{round_slug}__{home_slug}_v_{away_slug}.json
FILENAME_PATTERN = re.compile(
    r"^(\d{4})__(.+?)__(.+?)_v_(.+?)\.json$"
)

# All 59 player stat fields from the NRL API (union of 2013 and 2025 schemas).
# offsideWithinTenMetres was added post-2013; all others are present since 2013.
PLAYER_STAT_FIELDS: list[str] = [
    "allRunMetres",
    "allRuns",
    "bombKicks",
    "conversionAttempts",
    "conversions",
    "crossFieldKicks",
    "dummyHalfRunMetres",
    "dummyHalfRuns",
    "dummyPasses",
    "errors",
    "fantasyPointsTotal",
    "fieldGoals",
    "forcedDropOutKicks",
    "fortyTwentyKicks",
    "goalConversionRate",
    "goals",
    "grubberKicks",
    "handlingErrors",
    "hitUpRunMetres",
    "hitUps",
    "ineffectiveTackles",
    "intercepts",
    "kickMetres",
    "kickReturnMetres",
    "kicks",
    "kicksDead",
    "kicksDefused",
    "lineBreakAssists",
    "lineBreaks",
    "lineEngagedRuns",
    "minutesPlayed",
    "missedTackles",
    "offloads",
    "offsideWithinTenMetres",
    "onReport",
    "oneOnOneLost",
    "oneOnOneSteal",
    "onePointFieldGoals",
    "passes",
    "passesToRunRatio",
    "penalties",
    "penaltyGoals",
    "playTheBallAverageSpeed",
    "playTheBallTotal",
    "playerId",
    "points",
    "postContactMetres",
    "receipts",
    "ruckInfringements",
    "sendOffs",
    "sinBins",
    "stintOne",
    "tackleBreaks",
    "tackleEfficiency",
    "tacklesMade",
    "tries",
    "tryAssists",
    "twentyFortyKicks",
    "twoPointFieldGoals",
]

# Team-level stats from stats.groups that we want to extract.
# Maps the stat title in the JSON to our column name suffix.
# Includes 12 stats NOT extracted by scrape_nrl_match_stats.py:
#   Kick Return Metres, Average Play The Ball Speed, Receipts,
#   Total Passes, Dummy Passes, Kicks, Kick Defusal %, Bombs,
#   Grubbers, Ruck Infringements, Inside 10 Metres, Used (interchanges)
TEAM_STAT_MAP: dict[str, str] = {
    # -- Already extracted by scrape_nrl_match_stats.py --
    "Possession %": "possession_pct",
    "Time In Possession": "time_in_possession",
    "Completion Rate": "completion_rate",
    "All Runs": "all_runs",
    "All Run Metres": "run_metres",
    "Post Contact Metres": "post_contact_metres",
    "Line Breaks": "line_breaks",
    "Tackle Breaks": "tackle_breaks",
    "Average Set Distance": "avg_set_distance",
    "Offloads": "offloads",
    "Kicking Metres": "kick_metres",
    "Effective Tackle %": "effective_tackle_pct",
    "Tackles Made": "tackles_made",
    "Missed Tackles": "missed_tackles",
    "Ineffective Tackles": "ineffective_tackles",
    "Errors": "errors",
    "Penalties Conceded": "penalties_conceded",
    "Intercepts": "intercepts",
    "Forced Drop Outs": "forced_drop_outs",
    "On Reports": "on_reports",
    "Head Injury Assessment": "head_injury_assessment",
    # -- 12 NEW stats not previously extracted --
    "Kick Return Metres": "kick_return_metres",
    "Average Play The Ball Speed": "avg_play_the_ball_speed",
    "Receipts": "receipts",
    "Total Passes": "total_passes",
    "Dummy Passes": "dummy_passes",
    "Kicks": "kicks",
    "Kick Defusal %": "kick_defusal_pct",
    "Bombs": "bombs",
    "Grubbers": "grubbers",
    "Ruck Infringements": "ruck_infringements",
    "Inside 10 Metres": "inside_10_metres",
    "Used": "interchanges_used",
}

# Position classifications for player-quality features
SPINE_POSITIONS = {"Fullback", "Halfback", "Hooker", "Five-Eighth"}
FORWARD_POSITIONS = {"Prop", "Second Row", "Lock", "2nd Row"}


# ============================================================================
# Parsing helpers
# ============================================================================

def _parse_filename(filename: str) -> dict[str, str] | None:
    """Parse a JSON filename into year, round_slug, home_slug, away_slug.

    Returns None if the filename does not match the expected pattern.
    """
    m = FILENAME_PATTERN.match(filename)
    if m is None:
        return None
    return {
        "year": int(m.group(1)),
        "round_slug": m.group(2),
        "home_slug": m.group(3),
        "away_slug": m.group(4),
    }


def _slug_to_team(slug: str) -> str | None:
    """Convert an NRL URL slug to a canonical team name.

    Falls back to standardise_team_name if not in the lookup table.
    """
    canonical = SLUG_TO_CANONICAL.get(slug)
    if canonical:
        return canonical
    # Fallback: try the raw slug through standardise
    try:
        return standardise_team_name(slug)
    except KeyError:
        return None


def _parse_numeric(value: Any) -> float | None:
    """Parse a stat value to float, handling strings, commas, %, etc."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    s = s.replace(",", "")
    s = s.rstrip("%")
    s = s.rstrip("s")
    # Handle time format "32:08" -> seconds
    if re.match(r"^\d+:\d+$", s):
        parts = s.split(":")
        return float(parts[0]) * 60 + float(parts[1])
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _build_position_map(data: dict) -> dict[int, str]:
    """Build a mapping from playerId -> position using positionGroups.

    The positionGroups array provides the definitive starting position
    for each player. The homeTeam/awayTeam player lists also contain
    position info that we use as a fallback.
    """
    pos_map: dict[int, str] = {}

    # Primary: positionGroups (most reliable)
    for group in data.get("positionGroups", []):
        for pos_entry in group.get("positions", []):
            pos_name = pos_entry.get("name")
            if not pos_name:
                continue
            for id_key in ("homeProfileId", "awayProfileId"):
                pid = pos_entry.get(id_key)
                if pid is not None:
                    pos_map[int(pid)] = pos_name

    # Fallback: team player lists
    for team_key in ("homeTeam", "awayTeam"):
        team_data = data.get(team_key, {})
        for player in team_data.get("players", []):
            pid = player.get("playerId")
            pos = player.get("position")
            if pid is not None and pos is not None and int(pid) not in pos_map:
                pos_map[int(pid)] = pos

    return pos_map


def _build_player_info_map(data: dict) -> dict[int, dict]:
    """Build playerId -> {firstName, lastName, position, number} from team data."""
    info_map: dict[int, dict] = {}
    for team_key in ("homeTeam", "awayTeam"):
        team_data = data.get(team_key, {})
        for player in team_data.get("players", []):
            pid = player.get("playerId")
            if pid is not None:
                info_map[int(pid)] = {
                    "firstName": player.get("firstName", ""),
                    "lastName": player.get("lastName", ""),
                    "position": player.get("position"),
                    "number": player.get("number"),
                }
    return info_map


# ============================================================================
# Core extraction functions
# ============================================================================

def extract_match_metadata(
    data: dict,
    file_info: dict,
) -> dict:
    """Extract match-level metadata from a JSON file.

    Returns a single dict with match metadata fields.
    """
    # Referee: find official with position == "Referee"
    referee_name = None
    officials = data.get("officials") or []
    for official in officials:
        if official.get("position") == "Referee":
            first = official.get("firstName", "")
            last = official.get("lastName", "")
            referee_name = f"{first} {last}".strip()
            break

    home_slug = file_info["home_slug"]
    away_slug = file_info["away_slug"]
    home_team = _slug_to_team(home_slug)
    away_team = _slug_to_team(away_slug)

    # Try to get the API's team names if slug resolution fails
    if home_team is None:
        api_home = data.get("homeTeam", {}).get("name", "")
        if api_home:
            try:
                home_team = standardise_team_name(api_home)
            except KeyError:
                home_team = api_home
                logger.warning("Could not resolve home slug '%s' or API name '%s'",
                               home_slug, api_home)
    if away_team is None:
        api_away = data.get("awayTeam", {}).get("name", "")
        if api_away:
            try:
                away_team = standardise_team_name(api_away)
            except KeyError:
                away_team = api_away
                logger.warning("Could not resolve away slug '%s' or API name '%s'",
                               away_slug, api_away)

    # Home/away scores
    home_score = data.get("homeTeam", {}).get("score")
    away_score = data.get("awayTeam", {}).get("score")

    return {
        "year": file_info["year"],
        "round_slug": file_info["round_slug"],
        "home_slug": home_slug,
        "away_slug": away_slug,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "match_id": data.get("matchId"),
        "start_time": data.get("startTime"),
        "weather": data.get("weather"),
        "ground_conditions": data.get("groundConditions"),
        "referee_name": referee_name,
        "venue": data.get("venue"),
        "venue_city": data.get("venueCity"),
        "attendance": data.get("attendance"),
    }


def extract_player_stats(
    data: dict,
    file_info: dict,
) -> list[dict]:
    """Extract per-player stat rows from a JSON file.

    Returns a list of dicts, one per player per match.
    """
    rows: list[dict] = []
    stats = data.get("stats", {})
    players_data = stats.get("players", {})

    pos_map = _build_position_map(data)
    info_map = _build_player_info_map(data)

    home_slug = file_info["home_slug"]
    away_slug = file_info["away_slug"]
    home_team = _slug_to_team(home_slug)
    away_team = _slug_to_team(away_slug)

    # Fallback team names from API data
    if home_team is None:
        api_home = data.get("homeTeam", {}).get("name", "")
        try:
            home_team = standardise_team_name(api_home)
        except KeyError:
            home_team = api_home
    if away_team is None:
        api_away = data.get("awayTeam", {}).get("name", "")
        try:
            away_team = standardise_team_name(api_away)
        except KeyError:
            away_team = api_away

    for side_key, side_label, team_name in [
        ("homeTeam", "home", home_team),
        ("awayTeam", "away", away_team),
    ]:
        player_list = players_data.get(side_key, [])
        for player_stats in player_list:
            pid = player_stats.get("playerId")
            if pid is None:
                continue
            pid = int(pid)

            # Get player info from the team's player list (has names/position)
            info = info_map.get(pid, {})
            position = pos_map.get(pid, info.get("position"))

            row: dict[str, Any] = {
                "year": file_info["year"],
                "round_slug": file_info["round_slug"],
                "home_slug": home_slug,
                "away_slug": away_slug,
                "home_team": home_team,
                "away_team": away_team,
                "start_time": data.get("startTime"),
                "team": team_name,
                "side": side_label,
                "playerId": pid,
                "firstName": info.get("firstName", ""),
                "lastName": info.get("lastName", ""),
                "position": position,
                "number": info.get("number"),
            }

            # Extract all stat fields
            for field in PLAYER_STAT_FIELDS:
                if field == "playerId":
                    continue  # Already extracted above
                val = player_stats.get(field)
                if val is not None:
                    row[field] = _parse_numeric(val) if not isinstance(val, (int, float)) else val
                else:
                    row[field] = None

            rows.append(row)

    return rows


def extract_team_stats(
    data: dict,
    file_info: dict,
) -> dict:
    """Extract enhanced team-level stats from stats.groups.

    Returns a single dict with home_{stat} and away_{stat} columns for
    ALL available team stats including the 12 previously unextracted ones.
    """
    home_slug = file_info["home_slug"]
    away_slug = file_info["away_slug"]
    home_team = _slug_to_team(home_slug)
    away_team = _slug_to_team(away_slug)

    # Fallback team names from API data
    if home_team is None:
        api_home = data.get("homeTeam", {}).get("name", "")
        try:
            home_team = standardise_team_name(api_home)
        except KeyError:
            home_team = api_home
    if away_team is None:
        api_away = data.get("awayTeam", {}).get("name", "")
        try:
            away_team = standardise_team_name(api_away)
        except KeyError:
            away_team = api_away

    row: dict[str, Any] = {
        "year": file_info["year"],
        "round_slug": file_info["round_slug"],
        "home_slug": home_slug,
        "away_slug": away_slug,
        "home_team": home_team,
        "away_team": away_team,
        "start_time": data.get("startTime"),
    }

    # Extract from stats.groups
    stats = data.get("stats", {})
    groups = stats.get("groups", [])

    for group in groups:
        for stat in group.get("stats", []):
            title = stat.get("title", "")
            col_suffix = TEAM_STAT_MAP.get(title)
            if col_suffix is None:
                continue

            home_val = stat.get("homeValue", {})
            away_val = stat.get("awayValue", {})

            row[f"home_{col_suffix}"] = _parse_numeric(home_val.get("value"))
            row[f"away_{col_suffix}"] = _parse_numeric(away_val.get("value"))

            # For ratio stats (Completion Rate, Kick Defusal %), also store
            # numerator/denominator for more precise calculations
            if "numerator" in home_val:
                row[f"home_{col_suffix}_num"] = _parse_numeric(
                    home_val.get("numerator")
                )
                row[f"home_{col_suffix}_den"] = _parse_numeric(
                    home_val.get("denominator")
                )
            if "numerator" in away_val:
                row[f"away_{col_suffix}_num"] = _parse_numeric(
                    away_val.get("numerator")
                )
                row[f"away_{col_suffix}_den"] = _parse_numeric(
                    away_val.get("denominator")
                )

    return row


# ============================================================================
# Main extraction pipeline
# ============================================================================

def find_json_files() -> list[Path]:
    """Find all NRL API JSON files, sorted by name."""
    if not NRL_API_DIR.exists():
        raise FileNotFoundError(
            f"NRL API cache directory not found: {NRL_API_DIR}\n"
            f"Run scrape_nrl_match_stats.py first to populate the cache."
        )
    files = sorted(NRL_API_DIR.glob("*.json"))
    return files


def extract_all(
    json_files: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse all JSON files and return three DataFrames.

    Returns
    -------
    metadata_df : pd.DataFrame
        One row per match with weather, ground conditions, referee, venue, etc.
    player_df : pd.DataFrame
        One row per player per match with all 59 stat fields.
    team_stats_df : pd.DataFrame
        One row per match with ALL team-level stats from stats.groups.
    """
    metadata_rows: list[dict] = []
    player_rows: list[dict] = []
    team_stats_rows: list[dict] = []

    n_files = len(json_files)
    n_errors = 0
    n_no_players = 0

    for i, filepath in enumerate(json_files):
        if (i + 1) % 200 == 0 or i == 0:
            logger.info(
                "Processing file %d / %d (%.1f%%) ...",
                i + 1, n_files, (i + 1) / n_files * 100,
            )

        # Parse filename
        file_info = _parse_filename(filepath.name)
        if file_info is None:
            logger.warning("Skipping unrecognised filename: %s", filepath.name)
            n_errors += 1
            continue

        # Load JSON
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Error reading %s: %s", filepath.name, e)
            n_errors += 1
            continue

        # Skip empty/null responses
        if not data or not isinstance(data, dict):
            logger.warning("Empty or invalid JSON in %s", filepath.name)
            n_errors += 1
            continue

        # A. Match metadata
        try:
            meta = extract_match_metadata(data, file_info)
            metadata_rows.append(meta)
        except Exception as e:
            logger.warning("Error extracting metadata from %s: %s",
                           filepath.name, e)
            n_errors += 1

        # B. Player stats
        try:
            players = extract_player_stats(data, file_info)
            if players:
                player_rows.extend(players)
            else:
                n_no_players += 1
        except Exception as e:
            logger.warning("Error extracting player stats from %s: %s",
                           filepath.name, e)

        # C. Enhanced team stats
        try:
            team_row = extract_team_stats(data, file_info)
            team_stats_rows.append(team_row)
        except Exception as e:
            logger.warning("Error extracting team stats from %s: %s",
                           filepath.name, e)

    logger.info(
        "Extraction complete: %d files processed, %d errors, %d without player data",
        n_files, n_errors, n_no_players,
    )

    metadata_df = pd.DataFrame(metadata_rows)
    player_df = pd.DataFrame(player_rows)
    team_stats_df = pd.DataFrame(team_stats_rows)

    # Parse start_time to datetime for all DataFrames
    for df in (metadata_df, player_df, team_stats_df):
        if "start_time" in df.columns and not df.empty:
            df["start_time"] = pd.to_datetime(
                df["start_time"], errors="coerce", utc=True,
            )

    return metadata_df, player_df, team_stats_df


# ============================================================================
# Rolling player-quality features
# ============================================================================

def compute_player_quality_features(
    player_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling player-quality features aggregated per team per match.

    For each match, aggregates individual player stats into team-level
    player-quality indicators, then computes rolling averages to create
    pre-match features.

    Features computed:
      - team_avg_fantasy_{w}: rolling avg of team total fantasy points
      - spine_avg_fantasy_{w}: rolling avg of spine (FB+HB+HK+FE) fantasy
      - forward_run_metres_{w}: rolling avg of forwards total run metres
      - forward_tackles_{w}: rolling avg of forwards total tackles
      - team_experience: sum of minutesPlayed (current match, not rolling)
      - top_player_fantasy_{w}: rolling avg of top-3 player fantasy total

    Parameters
    ----------
    player_df : pd.DataFrame
        Player-level stats with columns: year, round_slug, home_slug,
        away_slug, team, side, position, fantasyPointsTotal, etc.
    windows : list of int, optional
        Rolling window sizes. Defaults to [5, 8].

    Returns
    -------
    pd.DataFrame
        One row per team per match with rolling player-quality features,
        plus match identifiers for merging.
    """
    if windows is None:
        windows = [5, 8]

    if player_df.empty:
        logger.warning("Empty player DataFrame, cannot compute quality features")
        return pd.DataFrame()

    df = player_df.copy()

    # Use start_time for temporal ordering
    if "start_time" in df.columns:
        df["date"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    else:
        # Fallback: construct approximate date from year + round
        df["date"] = pd.NaT

    # Create match key for grouping
    df["match_key"] = (
        df["year"].astype(str) + "__" + df["round_slug"] + "__"
        + df["home_slug"] + "__" + df["away_slug"]
    )

    # Normalise position names: "2nd Row" -> "Second Row"
    df["position_norm"] = df["position"].replace({"2nd Row": "Second Row"})

    # Classify players
    df["is_spine"] = df["position_norm"].isin(SPINE_POSITIONS)
    df["is_forward"] = df["position_norm"].isin(FORWARD_POSITIONS)

    # Ensure numeric stat columns
    for col in ["fantasyPointsTotal", "allRunMetres", "tacklesMade", "minutesPlayed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Aggregate per team per match
    # ------------------------------------------------------------------
    team_match_groups = df.groupby(["match_key", "team", "year", "round_slug",
                                     "home_slug", "away_slug", "side", "date"])

    agg_rows: list[dict] = []
    for (match_key, team, year, round_slug, home_slug, away_slug,
         side, date), group in team_match_groups:

        fantasy_total = group["fantasyPointsTotal"].sum()
        spine_fantasy = group.loc[group["is_spine"], "fantasyPointsTotal"].sum()
        forward_run_metres = group.loc[group["is_forward"], "allRunMetres"].sum()
        forward_tackles = group.loc[group["is_forward"], "tacklesMade"].sum()
        team_experience = group["minutesPlayed"].sum()

        # Top 3 fantasy scorers
        top3_fantasy = group.nlargest(3, "fantasyPointsTotal")["fantasyPointsTotal"].sum()

        agg_rows.append({
            "match_key": match_key,
            "team": team,
            "year": year,
            "round_slug": round_slug,
            "home_slug": home_slug,
            "away_slug": away_slug,
            "side": side,
            "date": date,
            "team_fantasy_total": fantasy_total,
            "spine_fantasy_total": spine_fantasy,
            "forward_run_metres_total": forward_run_metres,
            "forward_tackles_total": forward_tackles,
            "team_experience": team_experience,
            "top3_fantasy_total": top3_fantasy,
        })

    agg_df = pd.DataFrame(agg_rows)

    if agg_df.empty:
        logger.warning("No aggregated data, cannot compute rolling features")
        return pd.DataFrame()

    # Sort by team then date for proper rolling computation
    agg_df = agg_df.sort_values(["team", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Compute rolling averages (shifted to avoid lookahead)
    # ------------------------------------------------------------------
    stat_cols_to_roll = [
        ("team_fantasy_total", "team_avg_fantasy"),
        ("spine_fantasy_total", "spine_avg_fantasy"),
        ("forward_run_metres_total", "forward_run_metres"),
        ("forward_tackles_total", "forward_tackles"),
        ("top3_fantasy_total", "top_player_fantasy"),
    ]

    for window in windows:
        for src_col, dst_prefix in stat_cols_to_roll:
            col_name = f"{dst_prefix}_{window}"
            agg_df[col_name] = (
                agg_df.groupby("team")[src_col]
                .transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
            )

    # ------------------------------------------------------------------
    # Reshape: pivot from per-team to per-match with home_/away_ prefixes
    # ------------------------------------------------------------------
    home_df = agg_df[agg_df["side"] == "home"].copy()
    away_df = agg_df[agg_df["side"] == "away"].copy()

    # Columns to include in the final output
    feature_cols = ["team_experience"]
    for window in windows:
        for _, dst_prefix in stat_cols_to_roll:
            feature_cols.append(f"{dst_prefix}_{window}")

    merge_cols = ["year", "round_slug", "home_slug", "away_slug"]

    # Build home features
    home_features = home_df[merge_cols + feature_cols].copy()
    home_features = home_features.rename(
        columns={c: f"home_{c}" for c in feature_cols}
    )

    # Build away features
    away_features = away_df[merge_cols + feature_cols].copy()
    away_features = away_features.rename(
        columns={c: f"away_{c}" for c in feature_cols}
    )

    # Merge home and away
    result = home_features.merge(away_features, on=merge_cols, how="outer")

    # Add differentials
    for col in feature_cols:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        diff_col = f"diff_{col}"
        if home_col in result.columns and away_col in result.columns:
            result[diff_col] = result[home_col] - result[away_col]

    # Add team names for easier merging downstream
    slug_df = agg_df[
        agg_df["side"] == "home"
    ][merge_cols + ["team"]].drop_duplicates().rename(
        columns={"team": "home_team"}
    )
    result = result.merge(slug_df, on=merge_cols, how="left")

    slug_df_away = agg_df[
        agg_df["side"] == "away"
    ][merge_cols + ["team"]].drop_duplicates().rename(
        columns={"team": "away_team"}
    )
    result = result.merge(slug_df_away, on=merge_cols, how="left")

    return result


# ============================================================================
# Save outputs
# ============================================================================

def save_outputs(
    metadata_df: pd.DataFrame,
    player_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    quality_df: pd.DataFrame | None,
) -> dict[str, Path]:
    """Save all DataFrames to parquet files.

    Returns a dict mapping output name to file path.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if not metadata_df.empty:
        p = PROCESSED_DIR / "match_metadata.parquet"
        metadata_df.to_parquet(p, index=False, engine="pyarrow")
        paths["match_metadata"] = p
        logger.info("Saved %d rows to %s", len(metadata_df), p)

    if not player_df.empty:
        p = PROCESSED_DIR / "player_match_stats.parquet"
        player_df.to_parquet(p, index=False, engine="pyarrow")
        paths["player_match_stats"] = p
        logger.info("Saved %d rows to %s", len(player_df), p)

    if not team_stats_df.empty:
        p = PROCESSED_DIR / "match_stats_enhanced.parquet"
        team_stats_df.to_parquet(p, index=False, engine="pyarrow")
        paths["match_stats_enhanced"] = p
        logger.info("Saved %d rows to %s", len(team_stats_df), p)

    if quality_df is not None and not quality_df.empty:
        p = PROCESSED_DIR / "player_quality_features.parquet"
        quality_df.to_parquet(p, index=False, engine="pyarrow")
        paths["player_quality_features"] = p
        logger.info("Saved %d rows to %s", len(quality_df), p)

    return paths


# ============================================================================
# Summary statistics
# ============================================================================

def print_summary(
    metadata_df: pd.DataFrame,
    player_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    quality_df: pd.DataFrame | None,
    paths: dict[str, Path],
    elapsed: float,
) -> None:
    """Print a detailed summary of extracted data."""
    print()
    print("=" * 70)
    print("  NRL JSON Data Extraction - Summary")
    print("=" * 70)

    # -- Match metadata --
    print(f"\n  Match Metadata: {len(metadata_df)} matches")
    if not metadata_df.empty:
        years = sorted(metadata_df["year"].unique())
        print(f"  Years: {years[0]} - {years[-1]} ({len(years)} seasons)")
        print(f"  Matches per year:")
        for year, count in metadata_df.groupby("year").size().items():
            print(f"    {year}: {count}")

        # Weather coverage
        weather_non_null = metadata_df["weather"].notna().sum()
        ground_non_null = metadata_df["ground_conditions"].notna().sum()
        referee_non_null = metadata_df["referee_name"].notna().sum()
        attendance_non_null = metadata_df["attendance"].notna().sum()
        print(f"\n  Field coverage:")
        print(f"    weather:           {weather_non_null}/{len(metadata_df)} "
              f"({weather_non_null/len(metadata_df)*100:.1f}%)")
        print(f"    ground_conditions: {ground_non_null}/{len(metadata_df)} "
              f"({ground_non_null/len(metadata_df)*100:.1f}%)")
        print(f"    referee_name:      {referee_non_null}/{len(metadata_df)} "
              f"({referee_non_null/len(metadata_df)*100:.1f}%)")
        print(f"    attendance:        {attendance_non_null}/{len(metadata_df)} "
              f"({attendance_non_null/len(metadata_df)*100:.1f}%)")

        # Weather value counts
        if weather_non_null > 0:
            print(f"\n  Weather distribution:")
            for val, cnt in metadata_df["weather"].value_counts().head(10).items():
                print(f"    {val}: {cnt}")

        # Ground conditions distribution
        if ground_non_null > 0:
            print(f"\n  Ground conditions distribution:")
            for val, cnt in metadata_df["ground_conditions"].value_counts().head(10).items():
                print(f"    {val}: {cnt}")

    # -- Player stats --
    print(f"\n  Player Match Stats: {len(player_df)} rows")
    if not player_df.empty:
        n_players = player_df["playerId"].nunique()
        n_matches_with_players = player_df.groupby(
            ["year", "round_slug", "home_slug", "away_slug"]
        ).ngroups
        print(f"  Unique players: {n_players}")
        print(f"  Matches with player data: {n_matches_with_players}")
        avg_players = len(player_df) / n_matches_with_players
        print(f"  Avg players per match: {avg_players:.1f}")

        # Position coverage
        pos_non_null = player_df["position"].notna().sum()
        print(f"  Position coverage: {pos_non_null}/{len(player_df)} "
              f"({pos_non_null/len(player_df)*100:.1f}%)")

        if pos_non_null > 0:
            print(f"\n  Position distribution:")
            for val, cnt in player_df["position"].value_counts().head(15).items():
                print(f"    {val}: {cnt}")

        # Stat field coverage
        stat_fields_present = [
            f for f in PLAYER_STAT_FIELDS
            if f != "playerId" and f in player_df.columns
        ]
        print(f"\n  Player stat fields: {len(stat_fields_present)}")
        n_all_null = sum(
            1 for f in stat_fields_present if player_df[f].isna().all()
        )
        if n_all_null > 0:
            print(f"  Fields with no data: {n_all_null}")

    # -- Team stats --
    print(f"\n  Enhanced Team Stats: {len(team_stats_df)} matches")
    if not team_stats_df.empty:
        stat_cols = [c for c in team_stats_df.columns
                     if c.startswith("home_") or c.startswith("away_")]
        print(f"  Stat columns: {len(stat_cols)}")

        # Identify the 12 new stats and their coverage
        new_stat_suffixes = [
            "kick_return_metres", "avg_play_the_ball_speed", "receipts",
            "total_passes", "dummy_passes", "kicks", "kick_defusal_pct",
            "bombs", "grubbers", "ruck_infringements", "inside_10_metres",
            "interchanges_used",
        ]
        print(f"\n  New (previously unextracted) stat coverage:")
        for suffix in new_stat_suffixes:
            home_col = f"home_{suffix}"
            if home_col in team_stats_df.columns:
                non_null = team_stats_df[home_col].notna().sum()
                pct = non_null / len(team_stats_df) * 100
                print(f"    {suffix}: {non_null}/{len(team_stats_df)} ({pct:.1f}%)")
            else:
                print(f"    {suffix}: NOT FOUND in data")

    # -- Player quality features --
    if quality_df is not None and not quality_df.empty:
        print(f"\n  Player Quality Features: {len(quality_df)} match rows")
        feature_cols = [c for c in quality_df.columns
                        if c.startswith("home_") or c.startswith("away_")
                        or c.startswith("diff_")]
        print(f"  Feature columns: {len(feature_cols)}")

        # Show feature names
        home_features = sorted(
            c for c in quality_df.columns if c.startswith("home_")
            and c not in ("home_slug", "home_team")
        )
        print(f"\n  Home feature columns ({len(home_features)}):")
        for col in home_features:
            non_null = quality_df[col].notna().sum()
            print(f"    {col}: {non_null}/{len(quality_df)} non-null")

    # -- Output files --
    print(f"\n  Output files:")
    for name, path in paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"    {name}: {path.name} ({size_kb:.0f} KB)")

    print(f"\n  Completed in {elapsed:.1f}s")
    print("=" * 70)


# ============================================================================
# CLI entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract comprehensive data from cached NRL.com JSON match files. "
            "Produces match metadata, player stats, enhanced team stats, "
            "and rolling player-quality features."
        ),
    )
    parser.add_argument(
        "--skip-rolling",
        action="store_true",
        help="Skip computing rolling player-quality features",
    )
    parser.add_argument(
        "--rolling-windows",
        type=int,
        nargs="+",
        default=[5, 8],
        help="Rolling window sizes for player-quality features (default: 5 8)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Windows console encoding fix
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    t_start = time.time()

    print("=" * 70)
    print("  NRL JSON Data Extraction")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Find all JSON files
    # ------------------------------------------------------------------
    print(f"\n  Scanning {NRL_API_DIR} ...")
    json_files = find_json_files()
    print(f"  Found {len(json_files)} JSON files")

    if not json_files:
        print("  No JSON files found. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 2: Extract data from all files
    # ------------------------------------------------------------------
    print(f"\n  Extracting data from {len(json_files)} files ...")
    metadata_df, player_df, team_stats_df = extract_all(json_files)

    # ------------------------------------------------------------------
    # Step 3: Compute rolling player-quality features
    # ------------------------------------------------------------------
    quality_df = None
    if not args.skip_rolling and not player_df.empty:
        print(f"\n  Computing rolling player-quality features "
              f"(windows={args.rolling_windows}) ...")
        quality_df = compute_player_quality_features(
            player_df, windows=args.rolling_windows,
        )
        if quality_df is not None:
            print(f"  Computed {len(quality_df)} feature rows "
                  f"with {len(quality_df.columns)} columns")

    # ------------------------------------------------------------------
    # Step 4: Save outputs
    # ------------------------------------------------------------------
    print(f"\n  Saving outputs to {PROCESSED_DIR} ...")
    paths = save_outputs(metadata_df, player_df, team_stats_df, quality_df)

    # ------------------------------------------------------------------
    # Step 5: Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print_summary(
        metadata_df, player_df, team_stats_df, quality_df, paths, elapsed,
    )


if __name__ == "__main__":
    main()
