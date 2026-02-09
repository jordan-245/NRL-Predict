"""
Load and clean the AusSportsBetting historical NRL odds dataset.

The dataset is distributed as an Excel workbook downloadable from
`<https://www.aussportsbetting.com/data/historical-nrl-results-and-odds-data/>`_.

This module reads the Excel file with pandas, standardises team names via
:mod:`config.team_mappings` (if available), and returns a clean DataFrame
ready for joining with RLP match data.

Public API
----------
- :func:`load_odds` -- read and clean the Excel odds file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import RAW_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default file location
# ---------------------------------------------------------------------------
DEFAULT_ODDS_PATH: Path = RAW_DIR / "odds" / "nrl_odds.xlsx"

# ---------------------------------------------------------------------------
# Team-name standardisation
# ---------------------------------------------------------------------------
# The alias map is lazily loaded from config.team_mappings (if it exists).
# It maps every known alias to a single canonical team name.

_ALIAS_MAP: Optional[dict[str, str]] = None


def _build_alias_map() -> dict[str, str]:
    """Build a ``{alias_lower: canonical}`` mapping from the project
    configuration, or return an empty dict if no mappings are available."""
    try:
        from config.team_mappings import TEAM_ALIASES  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        logger.info(
            "config.team_mappings not found; team names will not be standardised."
        )
        return {}

    alias_map: dict[str, str] = {}
    for canonical, aliases in TEAM_ALIASES.items():
        alias_map[canonical.strip().lower()] = canonical
        for alias in aliases:
            alias_map[alias.strip().lower()] = canonical
    return alias_map


def _get_alias_map() -> dict[str, str]:
    """Return the cached alias map, building it on first access."""
    global _ALIAS_MAP  # noqa: PLW0603
    if _ALIAS_MAP is None:
        _ALIAS_MAP = _build_alias_map()
    return _ALIAS_MAP


def standardise_team_name(raw_name: str) -> str:
    """Map *raw_name* to its canonical form.

    If no mapping is found, the original name is returned with
    leading/trailing whitespace stripped.
    """
    alias_map = _get_alias_map()
    key = raw_name.strip().lower()
    return alias_map.get(key, raw_name.strip())


# ---------------------------------------------------------------------------
# Column name normalisation
# ---------------------------------------------------------------------------
# The Excel file's column names vary slightly between versions.  This map
# covers the common variants seen in files from 2013 onward.

_COLUMN_MAP: dict[str, str] = {
    # Date / time
    "date": "date",
    "kick-off (local)": "kickoff",
    "kick off (local)": "kickoff",
    "kickoff": "kickoff",
    "kick-off": "kickoff",
    # Teams
    "home team": "home_team",
    "home": "home_team",
    "away team": "away_team",
    "away": "away_team",
    # Venue
    "venue": "venue",
    # Scores
    "home score": "home_score",
    "home_score": "home_score",
    "away score": "away_score",
    "away_score": "away_score",
    # Playoff flag
    "play off game?": "is_playoff",
    "playoff": "is_playoff",
    "play off": "is_playoff",
    "finals": "is_playoff",
    # Over-time flag
    "over time?": "is_overtime",
    "overtime": "is_overtime",
    "overtime?": "is_overtime",
    # Head-to-head odds (closing / average)
    "home odds": "h2h_home",
    "home win": "h2h_home",
    "draw odds": "h2h_draw",
    "draw": "h2h_draw",
    "away odds": "h2h_away",
    "away win": "h2h_away",
    # Opening / closing / min / max -- H2H
    "home odds open": "h2h_home_open",
    "home open": "h2h_home_open",
    "home opening": "h2h_home_open",
    "home odds close": "h2h_home_close",
    "home close": "h2h_home_close",
    "home closing": "h2h_home_close",
    "home odds min": "h2h_home_min",
    "home min": "h2h_home_min",
    "home minimum": "h2h_home_min",
    "home odds max": "h2h_home_max",
    "home max": "h2h_home_max",
    "home maximum": "h2h_home_max",
    "away odds open": "h2h_away_open",
    "away open": "h2h_away_open",
    "away opening": "h2h_away_open",
    "away odds close": "h2h_away_close",
    "away close": "h2h_away_close",
    "away closing": "h2h_away_close",
    "away odds min": "h2h_away_min",
    "away min": "h2h_away_min",
    "away minimum": "h2h_away_min",
    "away odds max": "h2h_away_max",
    "away max": "h2h_away_max",
    "away maximum": "h2h_away_max",
    "draw open": "h2h_draw_open",
    "draw close": "h2h_draw_close",
    "draw min": "h2h_draw_min",
    "draw max": "h2h_draw_max",
    # Line / handicap (point spread)
    "home line": "line_home",
    "home line odds": "line_home_odds",
    "away line": "line_away",
    "away line odds": "line_away_odds",
    "home line open": "line_home_open",
    "home line close": "line_home_close",
    "home line min": "line_home_min",
    "home line max": "line_home_max",
    "away line open": "line_away_open",
    "away line close": "line_away_close",
    "away line min": "line_away_min",
    "away line max": "line_away_max",
    # Line odds (open / close / min / max)
    "home line odds open": "line_home_odds_open",
    "home line odds close": "line_home_odds_close",
    "home line odds min": "line_home_odds_min",
    "home line odds max": "line_home_odds_max",
    "away line odds open": "line_away_odds_open",
    "away line odds close": "line_away_odds_close",
    "away line odds min": "line_away_odds_min",
    "away line odds max": "line_away_odds_max",
    # Total (over/under)
    "total score over": "total_over",
    "total score under": "total_under",
    "over": "total_over",
    "under": "total_under",
    "total over": "total_over",
    "total under": "total_under",
    "total line": "total_line",
    "total score": "total_line",
    # Total score open / close / min / max (the line itself)
    "total score open": "total_line_open",
    "total score close": "total_line_close",
    "total score min": "total_line_min",
    "total score max": "total_line_max",
    # Total over/under odds (open / close / min / max)
    "total score over open": "total_over_open",
    "total score over close": "total_over_close",
    "total score over min": "total_over_min",
    "total score over max": "total_over_max",
    "total score under open": "total_under_open",
    "total score under close": "total_under_close",
    "total score under min": "total_under_min",
    "total score under max": "total_under_max",
    "total over open": "total_over_open",
    "total over close": "total_over_close",
    "total over min": "total_over_min",
    "total over max": "total_over_max",
    "total under open": "total_under_open",
    "total under close": "total_under_close",
    "total under min": "total_under_min",
    "total under max": "total_under_max",
    # Bookmakers count
    "bookmakers surveyed": "bookmakers_surveyed",
    "bookmakers": "bookmakers_surveyed",
    "# bookmakers": "bookmakers_surveyed",
    # Notes
    "notes": "notes",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns to canonical names."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_MAP:
            rename_map[col] = _COLUMN_MAP[key]
    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_odds(
    filepath: str | Path | None = None,
    *,
    standardise_teams: bool = True,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load the AusSportsBetting NRL odds Excel file.

    Parameters
    ----------
    filepath:
        Path to the ``.xlsx`` file.  Defaults to
        ``data/raw/odds/nrl.xlsx``.
    standardise_teams:
        If ``True``, apply :func:`standardise_team_name` to the home and
        away team columns.
    parse_dates:
        If ``True``, ensure the ``date`` column is parsed as
        ``datetime64``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardised column names and optionally
        standardised team names.  Rows are sorted by date ascending.

    Raises
    ------
    FileNotFoundError
        If the odds file does not exist at the given path.
    """
    path = Path(filepath) if filepath is not None else DEFAULT_ODDS_PATH

    if not path.is_file():
        raise FileNotFoundError(
            f"Odds file not found: {path}. "
            "Download it from https://www.aussportsbetting.com/data/"
            "historical-nrl-results-and-odds-data/ and place it at "
            f"{DEFAULT_ODDS_PATH}."
        )

    logger.info("Loading odds data from %s", path)

    # Read all sheets; the data is typically on the first sheet.
    # The AusSportsBetting file has a descriptive header in row 0
    # (source notes) with the actual column names in row 1, so we
    # use header=1.  If the first row looks like real column names
    # (no "Unnamed:" prefix), fall back to header=0.
    df = pd.read_excel(path, engine="openpyxl", header=1)
    # Guard: if header=1 produced mostly "Unnamed:" columns the file
    # probably has a single header row.  Re-read with header=0.
    unnamed_frac = sum(
        1 for c in df.columns if str(c).startswith("Unnamed:")
    ) / max(len(df.columns), 1)
    if unnamed_frac > 0.5:
        logger.debug("Re-reading with header=0 (single header row detected).")
        df = pd.read_excel(path, engine="openpyxl", header=0)

    # Normalise column names.
    df = _normalise_columns(df)

    # ---- Date parsing --------------------------------------------------
    if parse_dates and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # ---- Boolean flag helper ---------------------------------------------
    _TRUE_VALUES = {"Y", "YES", "1", "TRUE"}

    def _to_bool_flag(series: pd.Series) -> pd.Series:
        """Convert a Y/N/NaN column to a proper boolean Series."""
        return (
            series
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(_TRUE_VALUES)
        )

    # ---- Playoff flag --------------------------------------------------
    if "is_playoff" in df.columns:
        df["is_playoff"] = _to_bool_flag(df["is_playoff"])

    # ---- Overtime flag ---------------------------------------------------
    if "is_overtime" in df.columns:
        df["is_overtime"] = _to_bool_flag(df["is_overtime"])

    # ---- Score columns as int ------------------------------------------
    for col in ("home_score", "away_score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # ---- Standardise team names ----------------------------------------
    if standardise_teams:
        for col in ("home_team", "away_team"):
            if col in df.columns:
                df[col] = df[col].astype(str).apply(standardise_team_name)

    # ---- Derived columns -----------------------------------------------
    # Implied probabilities from closing H2H odds (1 / odds).
    for suffix in ("home", "away", "draw"):
        odds_col = f"h2h_{suffix}"
        prob_col = f"implied_prob_{suffix}"
        if odds_col in df.columns:
            df[prob_col] = (
                pd.to_numeric(df[odds_col], errors="coerce")
                .rdiv(1.0)
            )

    # Winner flag (binary target).
    if "home_score" in df.columns and "away_score" in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype("Int64")

    # ---- Sort and reset index ------------------------------------------
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    logger.info("Loaded %d rows from odds file.", len(df))
    return df
