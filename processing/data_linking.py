"""
Dataset linking / joining for the NRL prediction pipeline.

This module provides functions to join the cleaned datasets (matches, odds,
lineups, ladders, advanced stats) into a single master table suitable for
feature engineering.  Every join function reports unmatched records so that
data quality issues are surfaced early.

Key design decisions
--------------------
* Fuzzy date matching (+-1 day) is used when joining matches to odds because
  different sources sometimes record the match date differently (e.g.
  late-night Friday game recorded as Saturday by one source).
* All join functions return *left* joins anchored on the matches table, so
  that every match row is preserved even when supplementary data is missing.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===================================================================
# Helpers
# ===================================================================

def _report_unmatched(
    left: pd.DataFrame,
    merged: pd.DataFrame,
    indicator_col: str = "_merge",
    context: str = "",
) -> None:
    """Log a summary of unmatched rows from a merge with ``indicator=True``."""
    if indicator_col not in merged.columns:
        return
    counts = merged[indicator_col].value_counts()
    left_only = counts.get("left_only", 0)
    right_only = counts.get("right_only", 0)
    both = counts.get("both", 0)
    logger.info(
        "[%s] Matched: %d | Left-only (no match in right): %d | Right-only: %d",
        context or "merge",
        both,
        left_only,
        right_only,
    )
    if left_only > 0:
        unmatched = merged.loc[merged[indicator_col] == "left_only"]
        sample = unmatched.head(5)
        sample_cols = [
            c for c in ("date", "home_team", "away_team", "season", "round")
            if c in sample.columns
        ]
        if sample_cols:
            logger.debug(
                "[%s] Sample unmatched rows:\n%s",
                context,
                sample[sample_cols].to_string(index=False),
            )


def _normalise_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard join-key columns are present and consistently typed."""
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ("home_team", "away_team", "team"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    return out


# ===================================================================
# link_matches_odds  (fuzzy date matching +-1 day)
# ===================================================================

def link_matches_odds(
    matches_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    date_tolerance_days: int = 1,
) -> pd.DataFrame:
    """Join matches to odds on (home_team, away_team) with fuzzy date matching.

    Because data sources sometimes disagree on the exact date (e.g.
    Friday-night game logged as Saturday), this function tries an exact
    date match first, then falls back to a +-``date_tolerance_days`` window.

    Parameters
    ----------
    matches_df:
        Cleaned matches DataFrame.  Must have ``date``, ``home_team``,
        ``away_team``.
    odds_df:
        Cleaned odds DataFrame with the same key columns plus odds data.
    date_tolerance_days:
        Maximum number of days to search around the match date when an
        exact join fails.  Default is 1.

    Returns
    -------
    pd.DataFrame
        Left join of matches onto odds, with unmatched matches preserved
        (odds columns will be NaN for those rows).
    """
    matches = _normalise_join_keys(matches_df).copy()
    odds = _normalise_join_keys(odds_df).copy()

    # Suffix management: odds columns that collide with match columns
    odds_only_cols = [
        c for c in odds.columns if c not in ("date", "home_team", "away_team")
    ]

    # --- Pass 1: exact date match -----------------------------------------
    merged = matches.merge(
        odds,
        on=["date", "home_team", "away_team"],
        how="left",
        indicator=True,
        suffixes=("", "_odds"),
    )
    matched_mask = merged["_merge"] == "both"
    exact_matched = merged.loc[matched_mask].drop(columns=["_merge"])
    unmatched = merged.loc[~matched_mask].drop(
        columns=["_merge"] + [c + "_odds" for c in odds.columns if c + "_odds" in merged.columns]
    )
    # Also drop the odds-only columns that came through as NaN
    cols_to_drop = [c for c in odds_only_cols if c in unmatched.columns]
    unmatched = unmatched.drop(columns=cols_to_drop, errors="ignore")

    logger.info(
        "link_matches_odds pass 1 (exact date): %d matched, %d unmatched.",
        len(exact_matched),
        len(unmatched),
    )

    # --- Pass 2: fuzzy date match for remaining rows ----------------------
    if len(unmatched) > 0 and date_tolerance_days > 0:
        fuzzy_matched_parts = []
        still_unmatched_idx = []

        for idx, row in unmatched.iterrows():
            match_date = row.get("date")
            if pd.isna(match_date):
                still_unmatched_idx.append(idx)
                continue

            # Search within the tolerance window
            date_lo = match_date - timedelta(days=date_tolerance_days)
            date_hi = match_date + timedelta(days=date_tolerance_days)
            candidate = odds.loc[
                (odds["home_team"] == row["home_team"])
                & (odds["away_team"] == row["away_team"])
                & (odds["date"] >= date_lo)
                & (odds["date"] <= date_hi)
            ]
            if len(candidate) == 1:
                # Merge the single candidate
                odds_row = candidate.iloc[0]
                combined = row.copy()
                for c in odds_only_cols:
                    if c in odds_row.index:
                        combined[c] = odds_row[c]
                fuzzy_matched_parts.append(combined)
            elif len(candidate) > 1:
                # Pick the closest date
                candidate = candidate.copy()
                candidate["_date_diff"] = (
                    candidate["date"] - match_date
                ).abs()
                best = candidate.sort_values("_date_diff").iloc[0]
                combined = row.copy()
                for c in odds_only_cols:
                    if c in best.index:
                        combined[c] = best[c]
                fuzzy_matched_parts.append(combined)
            else:
                still_unmatched_idx.append(idx)

        n_fuzzy = len(fuzzy_matched_parts)
        logger.info(
            "link_matches_odds pass 2 (fuzzy ±%d day): %d matched, %d still unmatched.",
            date_tolerance_days,
            n_fuzzy,
            len(still_unmatched_idx),
        )

        # Rebuild final DataFrame
        parts = [exact_matched]
        if fuzzy_matched_parts:
            parts.append(pd.DataFrame(fuzzy_matched_parts))
        if still_unmatched_idx:
            parts.append(unmatched.loc[still_unmatched_idx])
        result = pd.concat(parts, ignore_index=True)
    else:
        result = merged.drop(columns=["_merge"], errors="ignore")

    # Sort chronologically
    if "date" in result.columns:
        result = result.sort_values("date").reset_index(drop=True)

    n_with_odds = result[odds_only_cols[0]].notna().sum() if odds_only_cols else 0
    logger.info(
        "link_matches_odds: %d total matches, %d with odds data.",
        len(result),
        n_with_odds,
    )
    return result


# ===================================================================
# link_matches_lineups
# ===================================================================

def link_matches_lineups(
    matches_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    join_keys: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Join match data with lineup data.

    If a ``match_id`` column exists in both DataFrames it is used as the
    join key.  Otherwise the join falls back to
    (``season``, ``round``, ``home_team``, ``away_team``).

    The result is a *long* table: one row per player-per-match, with match
    metadata attached.

    Parameters
    ----------
    matches_df:
        Cleaned matches DataFrame.
    lineups_df:
        Cleaned lineups DataFrame.
    join_keys:
        Explicit join columns.  Detected automatically if ``None``.

    Returns
    -------
    pd.DataFrame
        Left join of lineups onto matches.
    """
    matches = _normalise_join_keys(matches_df)
    lineups = _normalise_join_keys(lineups_df)

    if join_keys is None:
        if "match_id" in matches.columns and "match_id" in lineups.columns:
            join_keys = ["match_id"]
        else:
            # Determine the best available composite key
            possible_keys = ["season", "round", "home_team", "away_team"]
            join_keys = [
                k
                for k in possible_keys
                if k in matches.columns and k in lineups.columns
            ]
            if not join_keys:
                raise ValueError(
                    "Cannot determine join keys between matches and lineups. "
                    "Ensure both DataFrames share 'match_id' or "
                    "('season', 'round', 'home_team', 'away_team')."
                )

    merged = matches.merge(
        lineups,
        on=join_keys,
        how="left",
        indicator=True,
        suffixes=("", "_lineup"),
    )
    _report_unmatched(matches, merged, context="matches-lineups")
    merged = merged.drop(columns=["_merge"])

    logger.info(
        "link_matches_lineups: %d match rows x %d lineup rows -> %d joined rows.",
        len(matches),
        len(lineups),
        len(merged),
    )
    return merged


# ===================================================================
# link_matches_stats
# ===================================================================

def link_matches_stats(
    matches_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    join_keys: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Join match data with advanced match statistics.

    Parameters
    ----------
    matches_df:
        Cleaned matches DataFrame.
    stats_df:
        Advanced match stats DataFrame (e.g. possession %, completions,
        run metres, etc.).
    join_keys:
        Explicit join columns.  Detected automatically if ``None``.

    Returns
    -------
    pd.DataFrame
        Left join of matches onto stats (one row per match).
    """
    matches = _normalise_join_keys(matches_df)
    stats = _normalise_join_keys(stats_df)

    if join_keys is None:
        if "match_id" in matches.columns and "match_id" in stats.columns:
            join_keys = ["match_id"]
        else:
            possible_keys = ["season", "round", "home_team", "away_team"]
            join_keys = [
                k
                for k in possible_keys
                if k in matches.columns and k in stats.columns
            ]
            if not join_keys:
                raise ValueError(
                    "Cannot determine join keys between matches and stats."
                )

    merged = matches.merge(
        stats,
        on=join_keys,
        how="left",
        indicator=True,
        suffixes=("", "_stats"),
    )
    _report_unmatched(matches, merged, context="matches-stats")
    merged = merged.drop(columns=["_merge"])

    logger.info(
        "link_matches_stats: %d matches, %d with stats data.",
        len(merged),
        merged.iloc[:, -1].notna().sum(),
    )
    return merged


# ===================================================================
# build_master_dataset
# ===================================================================

def build_master_dataset(
    matches: pd.DataFrame,
    odds: Optional[pd.DataFrame] = None,
    lineups: Optional[pd.DataFrame] = None,
    ladders: Optional[pd.DataFrame] = None,
    stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the unified master table by joining all available datasets.

    The master table is anchored on the *matches* DataFrame.  Each
    supplementary dataset is left-joined in sequence.  The result is a
    match-level table (one row per match) with columns from every source.

    For lineups (which are player-level), this function aggregates to the
    match level before joining, producing columns like
    ``home_lineup_count`` and ``away_lineup_count``.

    Parameters
    ----------
    matches:
        Cleaned matches DataFrame (required).
    odds:
        Cleaned odds DataFrame (optional).
    lineups:
        Cleaned lineups DataFrame -- player-level rows (optional).
    ladders:
        Cleaned ladder DataFrame -- round-level standings (optional).
    stats:
        Cleaned advanced match stats DataFrame (optional).

    Returns
    -------
    pd.DataFrame
        Master dataset with one row per match and columns from all sources.
    """
    master = _normalise_join_keys(matches).copy()
    logger.info("build_master_dataset: starting with %d matches.", len(master))

    # --- 1. Odds ----------------------------------------------------------
    if odds is not None and len(odds) > 0:
        master = link_matches_odds(master, odds)
        logger.info("  + odds joined -> %d columns.", master.shape[1])

    # --- 2. Ladders (pre-match standings) ---------------------------------
    if ladders is not None and len(ladders) > 0:
        ladders_clean = _normalise_join_keys(ladders)

        # Join home-team ladder
        home_ladder = ladders_clean.copy()
        home_ladder = home_ladder.rename(
            columns={c: f"home_ladder_{c}" for c in home_ladder.columns
                     if c not in ("season", "round", "team")}
        )
        home_ladder = home_ladder.rename(columns={"team": "home_team"})

        home_join_keys = [
            k for k in ("season", "round", "home_team")
            if k in master.columns and k in home_ladder.columns
        ]
        if home_join_keys:
            master = master.merge(
                home_ladder,
                on=home_join_keys,
                how="left",
                suffixes=("", "_hl"),
            )

        # Join away-team ladder
        away_ladder = ladders_clean.copy()
        away_ladder = away_ladder.rename(
            columns={c: f"away_ladder_{c}" for c in away_ladder.columns
                     if c not in ("season", "round", "team")}
        )
        away_ladder = away_ladder.rename(columns={"team": "away_team"})

        away_join_keys = [
            k for k in ("season", "round", "away_team")
            if k in master.columns and k in away_ladder.columns
        ]
        if away_join_keys:
            master = master.merge(
                away_ladder,
                on=away_join_keys,
                how="left",
                suffixes=("", "_al"),
            )

        logger.info("  + ladders joined -> %d columns.", master.shape[1])

    # --- 3. Lineups (aggregate to match level) ----------------------------
    if lineups is not None and len(lineups) > 0:
        lineups_clean = _normalise_join_keys(lineups)

        # Determine grouping key
        if "match_id" in lineups_clean.columns:
            group_key = ["match_id", "team"]
        else:
            possible = ["season", "round", "home_team", "away_team", "team"]
            group_key = [k for k in possible if k in lineups_clean.columns]

        if group_key:
            lineup_agg = (
                lineups_clean.groupby(group_key, dropna=False)
                .agg(lineup_player_count=("player_name", "count"))
                .reset_index()
            )
            # Join via match_id or composite key
            join_keys = [k for k in group_key if k in master.columns]
            if join_keys:
                master = master.merge(
                    lineup_agg,
                    on=join_keys,
                    how="left",
                    suffixes=("", "_lu"),
                )

        logger.info("  + lineups joined -> %d columns.", master.shape[1])

    # --- 4. Advanced stats ------------------------------------------------
    if stats is not None and len(stats) > 0:
        master = link_matches_stats(master, stats)
        logger.info("  + stats joined -> %d columns.", master.shape[1])

    # --- Final sort -------------------------------------------------------
    if "date" in master.columns:
        master = master.sort_values("date").reset_index(drop=True)

    logger.info(
        "build_master_dataset: final shape %s (%d matches, %d features).",
        master.shape,
        master.shape[0],
        master.shape[1],
    )
    return master
