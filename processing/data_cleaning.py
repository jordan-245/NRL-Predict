"""
Data cleaning and standardisation for raw NRL datasets.

Each ``clean_*`` function takes a raw ``pandas.DataFrame`` (as produced by the
scraping layer) and returns a cleaned copy with:

* Standardised team names (via :func:`config.team_mappings.standardise_team_name`)
* Parsed and validated dates
* Null handling
* Validated numeric ranges
* Derived convenience columns

Edge cases handled: byes, draws, abandoned matches, walkovers.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.team_mappings import standardise_team_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VALID_POSITIONS = set(range(1, 18))  # Jersey numbers 1-17
_MIN_REASONABLE_ODDS = 1.01
_MAX_REASONABLE_ODDS = 101.0
_MAX_NRL_SCORE = 80  # Highest-ever NRL score is ~70-ish; generous cap

# Match result types that represent non-standard outcomes
_NON_STANDARD_RESULTS = {"bye", "abandoned", "cancelled", "walkover", "forfeit"}


# ===================================================================
# Helpers
# ===================================================================

def _safe_standardise(name: object) -> Optional[str]:
    """Attempt to standardise a team name; return ``None`` on failure."""
    if pd.isna(name) or not isinstance(name, str) or name.strip() == "":
        return None
    try:
        return standardise_team_name(name)
    except KeyError:
        logger.warning("Could not standardise team name: '%s'", name)
        return None


def _parse_date_column(series: pd.Series) -> pd.Series:
    """Coerce a Series to ``datetime64[ns]``, logging failures."""
    result = pd.to_datetime(series, errors="coerce", dayfirst=True)
    n_failed = result.isna().sum() - series.isna().sum()
    if n_failed > 0:
        logger.warning(
            "%d date values could not be parsed and were set to NaT.", n_failed
        )
    return result


# ===================================================================
# clean_matches
# ===================================================================

def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the raw matches table.

    Expected input columns (names are flexible; the function looks for common
    variants):

    * ``home_team``, ``away_team`` -- raw team names
    * ``home_score``, ``away_score`` -- numeric or string scores
    * ``date`` -- match date (string or datetime)
    * ``season`` / ``year`` -- season year
    * ``round`` -- round identifier (int or string for finals)
    * ``venue`` -- venue name (optional)
    * ``result_type`` -- (optional) e.g. ``"bye"``, ``"abandoned"``

    Returns
    -------
    pd.DataFrame
        Cleaned copy with additional columns:

        * ``home_team`` / ``away_team`` -- canonical names
        * ``date`` -- ``datetime64``
        * ``home_score`` / ``away_score`` -- validated ``Int64``
        * ``margin`` -- ``home_score - away_score``
        * ``home_win`` -- 1 / 0 / ``pd.NA`` (draw)
        * ``is_draw`` -- boolean
        * ``is_bye`` -- boolean flag
        * ``is_finals`` -- boolean flag
        * ``is_abandoned`` -- boolean flag
    """
    out = df.copy()

    # --- Team names -------------------------------------------------------
    for col in ("home_team", "away_team"):
        if col in out.columns:
            out[col] = out[col].map(_safe_standardise)

    n_unmapped_home = out["home_team"].isna().sum() if "home_team" in out.columns else 0
    n_unmapped_away = out["away_team"].isna().sum() if "away_team" in out.columns else 0
    if n_unmapped_home or n_unmapped_away:
        logger.warning(
            "Unmapped teams after standardisation: home=%d, away=%d",
            n_unmapped_home,
            n_unmapped_away,
        )

    # --- Date parsing -----------------------------------------------------
    if "date" in out.columns:
        out["date"] = _parse_date_column(out["date"])

    # --- Season / year ----------------------------------------------------
    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    elif "year" in out.columns:
        out["season"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    elif "date" in out.columns:
        out["season"] = out["date"].dt.year.astype("Int64")

    # --- Scores -----------------------------------------------------------
    for col in ("home_score", "away_score"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
            # Cap unreasonable values
            invalid_mask = (out[col] < 0) | (out[col] > _MAX_NRL_SCORE)
            n_invalid = invalid_mask.sum()
            if n_invalid > 0:
                logger.warning(
                    "%d rows with out-of-range %s (set to NA).", n_invalid, col
                )
                out.loc[invalid_mask, col] = pd.NA

    # --- Derived score columns -------------------------------------------
    if {"home_score", "away_score"}.issubset(out.columns):
        out["margin"] = out["home_score"] - out["away_score"]
        out["is_draw"] = (out["home_score"] == out["away_score"]) & out[
            "home_score"
        ].notna()
        out["home_win"] = pd.array(
            np.where(
                out["home_score"].isna() | out["away_score"].isna(),
                pd.NA,
                np.where(
                    out["home_score"] > out["away_score"],
                    1,
                    np.where(out["home_score"] < out["away_score"], 0, pd.NA),
                ),
            ),
            dtype="Int64",
        )
    else:
        out["margin"] = pd.NA
        out["is_draw"] = False
        out["home_win"] = pd.NA

    # --- Edge-case flags --------------------------------------------------
    result_col = "result_type" if "result_type" in out.columns else None

    out["is_bye"] = False
    out["is_abandoned"] = False

    if result_col:
        lower_result = out[result_col].astype(str).str.strip().str.lower()
        out["is_bye"] = lower_result == "bye"
        out["is_abandoned"] = lower_result.isin(
            {"abandoned", "cancelled", "walkover", "forfeit"}
        )

    # --- Finals flag ------------------------------------------------------
    if "round" in out.columns:
        round_str = out["round"].astype(str).str.strip().str.lower()
        finals_keywords = {"final", "qualif", "elim", "semi", "prelim", "grand"}
        out["is_finals"] = round_str.apply(
            lambda r: any(kw in r for kw in finals_keywords)
        )
    else:
        out["is_finals"] = False

    # --- Strip venue whitespace -------------------------------------------
    if "venue" in out.columns:
        out["venue"] = out["venue"].astype(str).str.strip()

    logger.info(
        "clean_matches: %d rows in, %d rows out (byes=%d, abandoned=%d, draws=%d).",
        len(df),
        len(out),
        out["is_bye"].sum(),
        out["is_abandoned"].sum(),
        out["is_draw"].sum(),
    )
    return out


# ===================================================================
# clean_lineups
# ===================================================================

def clean_lineups(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate team lineup data.

    Expected columns:

    * ``match_id`` or (``season``, ``round``, ``home_team``, ``away_team``)
    * ``team`` -- the team this player belongs to
    * ``player_name`` -- player full name
    * ``position`` -- jersey number (1--17)

    Returns
    -------
    pd.DataFrame
        Cleaned copy with validated positions and standardised names.
    """
    out = df.copy()

    # --- Team name standardisation ----------------------------------------
    if "team" in out.columns:
        out["team"] = out["team"].map(_safe_standardise)

    for col in ("home_team", "away_team"):
        if col in out.columns:
            out[col] = out[col].map(_safe_standardise)

    # --- Player names -----------------------------------------------------
    if "player_name" in out.columns:
        # Strip whitespace, collapse internal whitespace, title-case
        out["player_name"] = (
            out["player_name"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        # Flag missing / placeholder names
        missing_mask = out["player_name"].isin({"", "nan", "None", "TBC", "TBA"})
        if missing_mask.any():
            logger.warning(
                "%d lineup rows have missing/placeholder player names.",
                missing_mask.sum(),
            )
            out.loc[missing_mask, "player_name"] = pd.NA

    # --- Position validation ----------------------------------------------
    if "position" in out.columns:
        out["position"] = pd.to_numeric(out["position"], errors="coerce").astype(
            "Int64"
        )
        invalid_pos = ~out["position"].isin(_VALID_POSITIONS) & out[
            "position"
        ].notna()
        if invalid_pos.any():
            logger.warning(
                "%d lineup rows have invalid position numbers (outside 1-17).",
                invalid_pos.sum(),
            )
            out.loc[invalid_pos, "position"] = pd.NA

    # --- Duplicate detection (same player listed twice for same match) ----
    dedup_cols = [
        c
        for c in ("match_id", "team", "player_name")
        if c in out.columns
    ]
    if len(dedup_cols) == 3:
        n_before = len(out)
        out = out.drop_duplicates(subset=dedup_cols, keep="first")
        n_dropped = n_before - len(out)
        if n_dropped:
            logger.warning("Dropped %d duplicate lineup entries.", n_dropped)

    # --- Date parsing (if present) ----------------------------------------
    if "date" in out.columns:
        out["date"] = _parse_date_column(out["date"])

    logger.info("clean_lineups: %d rows in, %d rows out.", len(df), len(out))
    return out


# ===================================================================
# clean_odds
# ===================================================================

def clean_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Clean bookmaker odds data and derive implied probabilities.

    Expected columns:

    * ``home_team``, ``away_team``
    * ``date``
    * ``home_odds``, ``away_odds`` -- decimal odds (>= 1.0)
    * Optionally: ``draw_odds``, ``home_odds_open``, ``home_odds_close``, etc.

    Returns
    -------
    pd.DataFrame
        Cleaned copy with added columns:

        * ``home_implied_prob`` -- 1 / home_odds (raw, before overround removal)
        * ``away_implied_prob`` -- 1 / away_odds
        * ``overround`` -- sum of implied probs (>1.0 indicates bookmaker margin)
        * ``home_implied_prob_fair`` -- after removing overround
        * ``away_implied_prob_fair``
    """
    out = df.copy()

    # --- Team name standardisation ----------------------------------------
    for col in ("home_team", "away_team"):
        if col in out.columns:
            out[col] = out[col].map(_safe_standardise)

    # --- Date parsing -----------------------------------------------------
    if "date" in out.columns:
        out["date"] = _parse_date_column(out["date"])

    # --- Validate and clean odds columns ----------------------------------
    odds_cols = [c for c in out.columns if "odds" in c.lower()]
    for col in odds_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        invalid_mask = (out[col] < _MIN_REASONABLE_ODDS) | (
            out[col] > _MAX_REASONABLE_ODDS
        )
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            logger.warning(
                "%d values in '%s' outside [%.2f, %.1f] (set to NaN).",
                n_invalid,
                col,
                _MIN_REASONABLE_ODDS,
                _MAX_REASONABLE_ODDS,
            )
            out.loc[invalid_mask, col] = np.nan

    # --- Implied probabilities (raw) --------------------------------------
    if "home_odds" in out.columns:
        out["home_implied_prob"] = 1.0 / out["home_odds"]
    if "away_odds" in out.columns:
        out["away_implied_prob"] = 1.0 / out["away_odds"]

    # --- Overround and fair probabilities ---------------------------------
    prob_cols_present = {"home_implied_prob", "away_implied_prob"}.issubset(
        out.columns
    )
    if prob_cols_present:
        draw_prob = np.float64(0.0)
        if "draw_odds" in out.columns:
            draw_prob = (1.0 / out["draw_odds"]).fillna(0.0)

        out["overround"] = (
            out["home_implied_prob"] + out["away_implied_prob"] + draw_prob
        )
        out["home_implied_prob_fair"] = out["home_implied_prob"] / out["overround"]
        out["away_implied_prob_fair"] = out["away_implied_prob"] / out["overround"]

    # --- Opening-odds implied probabilities (if available) ----------------
    if "home_odds_open" in out.columns:
        out["home_open_implied_prob"] = 1.0 / out["home_odds_open"]
    if "away_odds_open" in out.columns:
        out["away_open_implied_prob"] = 1.0 / out["away_odds_open"]

    logger.info("clean_odds: %d rows in, %d rows out.", len(df), len(out))
    return out


# ===================================================================
# clean_ladder
# ===================================================================

def clean_ladder(df: pd.DataFrame) -> pd.DataFrame:
    """Clean round-by-round ladder (standings) data and compute derived fields.

    Expected columns:

    * ``season`` / ``year``
    * ``round`` -- round number
    * ``team`` -- team name
    * ``played`` (P), ``won`` (W), ``drawn`` (D), ``lost`` (L)
    * ``points_for`` (F), ``points_against`` (A)
    * ``points`` (Pts) -- competition points
    * ``position`` -- ladder position (optional; can be recomputed)

    Returns
    -------
    pd.DataFrame
        Cleaned copy with added derived columns:

        * ``win_pct`` -- won / played
        * ``for_against_ratio`` -- points_for / points_against
        * ``point_diff`` -- points_for - points_against
        * ``point_diff_per_game`` -- point_diff / played
    """
    out = df.copy()

    # --- Team names -------------------------------------------------------
    if "team" in out.columns:
        out["team"] = out["team"].map(_safe_standardise)

    # --- Season -----------------------------------------------------------
    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    elif "year" in out.columns:
        out["season"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

    # --- Round ------------------------------------------------------------
    if "round" in out.columns:
        out["round"] = pd.to_numeric(out["round"], errors="coerce").astype("Int64")

    # --- Numeric standings columns ----------------------------------------
    standings_cols = [
        "played",
        "won",
        "drawn",
        "lost",
        "points_for",
        "points_against",
        "points",
        "position",
    ]
    for col in standings_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    # Map common abbreviations to standard names
    _col_renames = {
        "P": "played",
        "W": "won",
        "D": "drawn",
        "L": "lost",
        "F": "points_for",
        "A": "points_against",
        "Pts": "points",
        "Pos": "position",
    }
    existing_renames = {k: v for k, v in _col_renames.items() if k in out.columns}
    if existing_renames:
        out = out.rename(columns=existing_renames)

    # --- Validate non-negative standings ----------------------------------
    for col in ("played", "won", "drawn", "lost", "points_for", "points_against"):
        if col in out.columns:
            neg_mask = out[col] < 0
            if neg_mask.any():
                logger.warning(
                    "%d negative values in '%s' (set to NA).", neg_mask.sum(), col
                )
                out.loc[neg_mask, col] = pd.NA

    # --- Validate W + D + L == P (where all are present) ------------------
    if {"played", "won", "drawn", "lost"}.issubset(out.columns):
        computed_total = out["won"] + out["drawn"] + out["lost"]
        mismatch = (computed_total != out["played"]) & out["played"].notna()
        if mismatch.any():
            logger.warning(
                "%d rows where W+D+L != P; trusting individual columns.",
                mismatch.sum(),
            )

    # --- Derived fields ---------------------------------------------------
    if "played" in out.columns and "won" in out.columns:
        played_safe = out["played"].replace(0, pd.NA).astype("Float64")
        out["win_pct"] = out["won"].astype("Float64") / played_safe

    if "points_for" in out.columns and "points_against" in out.columns:
        out["point_diff"] = out["points_for"] - out["points_against"]

        pa_safe = out["points_against"].replace(0, pd.NA).astype("Float64")
        out["for_against_ratio"] = out["points_for"].astype("Float64") / pa_safe

        if "played" in out.columns:
            played_safe = out["played"].replace(0, pd.NA).astype("Float64")
            out["point_diff_per_game"] = (
                out["point_diff"].astype("Float64") / played_safe
            )

    logger.info("clean_ladder: %d rows in, %d rows out.", len(df), len(out))
    return out
