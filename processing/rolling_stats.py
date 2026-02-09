"""
Rolling-window aggregations for NRL team form and head-to-head features.

Every function in this module computes features strictly from data
**before** each match (no leakage).  The approach is:

1. Build a per-team match history sorted chronologically.
2. For each match, look back over the specified window of *completed*
   games and aggregate.
3. Attach the result to the original match row.

Season boundaries are handled explicitly: by default, rolling windows
can span across seasons (a team's form at the end of season N carries
into the start of season N+1).  An optional ``respect_season`` flag
restricts windows to the current season only.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===================================================================
# Internal helpers
# ===================================================================

def _build_team_match_log(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the matches table into a *team-centric* match log.

    Each row in the output represents a single team's participation in a
    single match, with columns normalised to that team's perspective:

    * ``team``, ``opponent``
    * ``is_home``
    * ``points_for``, ``points_against``, ``margin``
    * ``win`` (1/0/0.5 for draw)
    * ``date``, ``season``, ``round``, ``match_idx`` (row position in
      the original DF)

    The output is sorted by (team, date, match_idx).
    """
    required = {"home_team", "away_team", "home_score", "away_score", "date"}
    missing = required - set(matches_df.columns)
    if missing:
        raise ValueError(
            f"matches_df is missing required columns: {missing}"
        )

    df = matches_df.copy()
    df["_match_idx"] = range(len(df))

    # Home perspective
    home = pd.DataFrame(
        {
            "match_idx": df["_match_idx"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "is_home": True,
            "points_for": df["home_score"],
            "points_against": df["away_score"],
            "date": df["date"],
            "season": df.get("season"),
            "round": df.get("round"),
        }
    )

    # Away perspective
    away = pd.DataFrame(
        {
            "match_idx": df["_match_idx"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "is_home": False,
            "points_for": df["away_score"],
            "points_against": df["home_score"],
            "date": df["date"],
            "season": df.get("season"),
            "round": df.get("round"),
        }
    )

    log = pd.concat([home, away], ignore_index=True)
    log["margin"] = log["points_for"] - log["points_against"]

    # Win indicator (1 = win, 0.5 = draw, 0 = loss)
    log["win"] = np.where(
        log["margin"] > 0,
        1.0,
        np.where(log["margin"] == 0, 0.5, 0.0),
    )

    log = log.sort_values(["team", "date", "match_idx"]).reset_index(drop=True)
    return log


def _rolling_agg_for_team(
    team_log: pd.DataFrame,
    window: int,
    respect_season: bool = False,
) -> pd.DataFrame:
    """Compute rolling aggregates for a single team's match log.

    Returns one row per match with columns:
    ``win_rate_{w}``, ``avg_pf_{w}``, ``avg_pa_{w}``, ``avg_margin_{w}``
    where ``w`` is the window size.

    Only completed (non-NaN score) matches count toward the window.
    """
    w = window
    cols = {
        f"win_rate_{w}": [],
        f"avg_pf_{w}": [],
        f"avg_pa_{w}": [],
        f"avg_margin_{w}": [],
    }
    match_indices = []

    valid = team_log.dropna(subset=["points_for", "points_against"]).copy()
    valid_dates = valid["date"].values
    valid_seasons = valid["season"].values if "season" in valid.columns else None

    for i, (_, row) in enumerate(team_log.iterrows()):
        match_indices.append(row["match_idx"])

        # Matches strictly BEFORE this one (by position in the sorted log)
        if i == 0:
            for col in cols:
                cols[col].append(np.nan)
            continue

        # Find completed prior matches
        prior = valid.loc[valid.index < row.name]

        if respect_season and "season" in team_log.columns:
            prior = prior.loc[prior["season"] == row.get("season")]

        # Take the last `window` matches
        prior_window = prior.tail(window)

        if len(prior_window) == 0:
            for col in cols:
                cols[col].append(np.nan)
        else:
            cols[f"win_rate_{w}"].append(prior_window["win"].mean())
            cols[f"avg_pf_{w}"].append(prior_window["points_for"].mean())
            cols[f"avg_pa_{w}"].append(prior_window["points_against"].mean())
            cols[f"avg_margin_{w}"].append(prior_window["margin"].mean())

    result = pd.DataFrame(cols)
    result["match_idx"] = match_indices
    return result


# ===================================================================
# compute_rolling_form
# ===================================================================

def compute_rolling_form(
    matches_df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    respect_season: bool = False,
) -> pd.DataFrame:
    """Compute rolling form features for every team before every match.

    For each team and each window size, the following features are computed
    using only matches that occurred **before** the current one:

    * ``{side}_win_rate_{w}`` -- fraction of wins in last *w* games
    * ``{side}_avg_pf_{w}`` -- average points for
    * ``{side}_avg_pa_{w}`` -- average points against
    * ``{side}_avg_margin_{w}`` -- average margin

    where ``{side}`` is ``home`` or ``away`` and ``{w}`` is the window.

    Parameters
    ----------
    matches_df:
        Cleaned, chronologically sorted matches DataFrame.
    windows:
        List of lookback window sizes.  Default ``[3, 5, 8, 10]``.
    respect_season:
        If True, rolling windows do not cross season boundaries.

    Returns
    -------
    pd.DataFrame
        Copy of *matches_df* with rolling-form columns appended.
    """
    if windows is None:
        windows = [3, 5, 8, 10]

    df = matches_df.copy()
    df["_match_idx"] = range(len(df))

    team_log = _build_team_match_log(df)

    # Pre-compute rolling stats per team per window
    all_teams = team_log["team"].unique()
    team_results: dict[str, dict[int, pd.DataFrame]] = {}

    for team in all_teams:
        t_log = team_log.loc[team_log["team"] == team].copy()
        team_results[team] = {}
        for w in windows:
            team_results[team][w] = _rolling_agg_for_team(
                t_log, w, respect_season=respect_season
            )

    # Build lookup: (team, match_idx) -> {feature: value}
    lookup: dict[tuple[str, int], dict[str, float]] = {}
    for team in all_teams:
        for w in windows:
            res = team_results[team][w]
            feat_cols = [c for c in res.columns if c != "match_idx"]
            for _, row in res.iterrows():
                key = (team, int(row["match_idx"]))
                if key not in lookup:
                    lookup[key] = {}
                for c in feat_cols:
                    lookup[key][c] = row[c]

    # Attach to original DataFrame as home_ and away_ prefixed columns
    new_cols: dict[str, list] = {}
    for w in windows:
        for stat in ("win_rate", "avg_pf", "avg_pa", "avg_margin"):
            new_cols[f"home_{stat}_{w}"] = []
            new_cols[f"away_{stat}_{w}"] = []

    for _, row in df.iterrows():
        idx = row["_match_idx"]
        home = row["home_team"]
        away = row["away_team"]
        home_feats = lookup.get((home, idx), {})
        away_feats = lookup.get((away, idx), {})

        for w in windows:
            for stat in ("win_rate", "avg_pf", "avg_pa", "avg_margin"):
                key_name = f"{stat}_{w}"
                new_cols[f"home_{stat}_{w}"].append(home_feats.get(key_name, np.nan))
                new_cols[f"away_{stat}_{w}"].append(away_feats.get(key_name, np.nan))

    for col_name, values in new_cols.items():
        df[col_name] = values

    df = df.drop(columns=["_match_idx"])

    logger.info(
        "compute_rolling_form: added %d rolling columns for windows %s.",
        len(new_cols),
        windows,
    )
    return df


# ===================================================================
# compute_h2h_features
# ===================================================================

def compute_h2h_features(
    matches_df: pd.DataFrame,
    lookbacks: Optional[List[Union[int, str]]] = None,
) -> pd.DataFrame:
    """Compute head-to-head features between specific team pairs.

    For each match, computes features based on *prior* meetings between
    the same two teams:

    * ``h2h_home_win_rate_{lb}`` -- home team's win rate against the away
      team over the lookback window
    * ``h2h_avg_margin_{lb}`` -- average margin (from home perspective)
      in recent head-to-head meetings
    * ``h2h_matches_{lb}`` -- number of H2H meetings in the window

    Parameters
    ----------
    matches_df:
        Cleaned, chronologically sorted matches DataFrame.
    lookbacks:
        List of lookback specifications.  An integer means "last N meetings".
        A string like ``"3years"`` means "meetings in the last 3 calendar
        years".  Default ``[3, 5, "3years"]``.

    Returns
    -------
    pd.DataFrame
        Copy of *matches_df* with H2H columns appended.
    """
    if lookbacks is None:
        lookbacks = [3, 5, "3years"]

    df = matches_df.copy()
    df["_match_idx"] = range(len(df))

    # Pre-sort
    df = df.sort_values("date").reset_index(drop=True)

    # Build H2H features
    h2h_cols: dict[str, list] = {}
    for lb in lookbacks:
        lb_str = str(lb)
        h2h_cols[f"h2h_home_win_rate_{lb_str}"] = []
        h2h_cols[f"h2h_avg_margin_{lb_str}"] = []
        h2h_cols[f"h2h_matches_{lb_str}"] = []

    for i, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        match_date = row["date"]

        # All prior meetings between these two teams (either as home or away)
        prior = df.loc[:i - 1] if i > 0 else df.iloc[0:0]
        h2h_prior = prior.loc[
            ((prior["home_team"] == home) & (prior["away_team"] == away))
            | ((prior["home_team"] == away) & (prior["away_team"] == home))
        ].copy()

        # Filter out rows with missing scores
        h2h_prior = h2h_prior.dropna(subset=["home_score", "away_score"])

        for lb in lookbacks:
            lb_str = str(lb)

            if isinstance(lb, int):
                window = h2h_prior.tail(lb)
            elif isinstance(lb, str) and lb.endswith("years"):
                try:
                    n_years = int(lb.replace("years", ""))
                except ValueError:
                    n_years = 3
                if pd.notna(match_date):
                    cutoff = match_date - pd.DateOffset(years=n_years)
                    window = h2h_prior.loc[h2h_prior["date"] >= cutoff]
                else:
                    window = h2h_prior
            else:
                window = h2h_prior

            if len(window) == 0:
                h2h_cols[f"h2h_home_win_rate_{lb_str}"].append(np.nan)
                h2h_cols[f"h2h_avg_margin_{lb_str}"].append(np.nan)
                h2h_cols[f"h2h_matches_{lb_str}"].append(0)
                continue

            # Compute margin from the current home team's perspective
            # When current home was historical home: margin = home_score - away_score
            # When current home was historical away: margin = away_score - home_score
            margins = []
            wins = []
            for _, h2h_row in window.iterrows():
                if h2h_row["home_team"] == home:
                    m = h2h_row["home_score"] - h2h_row["away_score"]
                else:
                    m = h2h_row["away_score"] - h2h_row["home_score"]
                margins.append(m)
                wins.append(1.0 if m > 0 else (0.5 if m == 0 else 0.0))

            h2h_cols[f"h2h_home_win_rate_{lb_str}"].append(np.mean(wins))
            h2h_cols[f"h2h_avg_margin_{lb_str}"].append(np.mean(margins))
            h2h_cols[f"h2h_matches_{lb_str}"].append(len(window))

    for col_name, values in h2h_cols.items():
        df[col_name] = values

    df = df.drop(columns=["_match_idx"])

    logger.info(
        "compute_h2h_features: added %d H2H columns for lookbacks %s.",
        len(h2h_cols),
        lookbacks,
    )
    return df


# ===================================================================
# compute_exponential_weighted
# ===================================================================

def compute_exponential_weighted(
    matches_df: pd.DataFrame,
    half_life: int = 5,
    respect_season: bool = False,
) -> pd.DataFrame:
    """Compute exponentially weighted moving average (EWMA) form features.

    Recent matches count more than older ones.  The *half_life* parameter
    controls how quickly the weight decays: after ``half_life`` matches the
    weight is 0.5x the most recent match's weight.

    Features produced (per side):

    * ``{side}_ewma_win_{hl}`` -- EWMA of the win indicator
    * ``{side}_ewma_margin_{hl}`` -- EWMA of the scoring margin
    * ``{side}_ewma_pf_{hl}`` -- EWMA of points for
    * ``{side}_ewma_pa_{hl}`` -- EWMA of points against

    where ``{hl}`` is the half-life value.

    Parameters
    ----------
    matches_df:
        Cleaned, chronologically sorted matches DataFrame.
    half_life:
        EWMA half-life in number of games.  Default 5.
    respect_season:
        If True, EWMA resets at season boundaries.

    Returns
    -------
    pd.DataFrame
        Copy of *matches_df* with EWMA columns appended.
    """
    df = matches_df.copy()
    df["_match_idx"] = range(len(df))

    team_log = _build_team_match_log(df)
    alpha = 1 - np.exp(-np.log(2) / half_life)
    hl_str = str(half_life)

    # Compute EWMA per team
    # Store: (team, match_idx) -> {feature: value}
    ewma_lookup: dict[tuple[str, int], dict[str, float]] = {}

    for team in team_log["team"].unique():
        t_log = team_log.loc[team_log["team"] == team].copy()

        # Running EWMA state
        ewma_win = np.nan
        ewma_margin = np.nan
        ewma_pf = np.nan
        ewma_pa = np.nan
        prev_season = None

        for _, row in t_log.iterrows():
            midx = int(row["match_idx"])
            cur_season = row.get("season")

            # Reset at season boundary if requested
            if respect_season and prev_season is not None and cur_season != prev_season:
                ewma_win = np.nan
                ewma_margin = np.nan
                ewma_pf = np.nan
                ewma_pa = np.nan

            # Store the PRE-match EWMA (i.e. before this game is included)
            ewma_lookup[(team, midx)] = {
                f"ewma_win_{hl_str}": ewma_win,
                f"ewma_margin_{hl_str}": ewma_margin,
                f"ewma_pf_{hl_str}": ewma_pf,
                f"ewma_pa_{hl_str}": ewma_pa,
            }

            # Update EWMA with this match's data
            if pd.notna(row["points_for"]) and pd.notna(row["points_against"]):
                w = row["win"]
                m = row["margin"]
                pf = row["points_for"]
                pa = row["points_against"]

                if np.isnan(ewma_win):
                    # First valid match: initialise
                    ewma_win = w
                    ewma_margin = m
                    ewma_pf = pf
                    ewma_pa = pa
                else:
                    ewma_win = alpha * w + (1 - alpha) * ewma_win
                    ewma_margin = alpha * m + (1 - alpha) * ewma_margin
                    ewma_pf = alpha * pf + (1 - alpha) * ewma_pf
                    ewma_pa = alpha * pa + (1 - alpha) * ewma_pa

            prev_season = cur_season

    # Attach to original DataFrame
    feat_names = [
        f"ewma_win_{hl_str}",
        f"ewma_margin_{hl_str}",
        f"ewma_pf_{hl_str}",
        f"ewma_pa_{hl_str}",
    ]
    new_cols: dict[str, list] = {}
    for side in ("home", "away"):
        for f in feat_names:
            new_cols[f"{side}_{f}"] = []

    for _, row in df.iterrows():
        idx = row["_match_idx"]
        home_feats = ewma_lookup.get((row["home_team"], idx), {})
        away_feats = ewma_lookup.get((row["away_team"], idx), {})
        for f in feat_names:
            new_cols[f"home_{f}"].append(home_feats.get(f, np.nan))
            new_cols[f"away_{f}"].append(away_feats.get(f, np.nan))

    for col_name, values in new_cols.items():
        df[col_name] = values

    df = df.drop(columns=["_match_idx"])

    logger.info(
        "compute_exponential_weighted: added %d EWMA columns (half_life=%d).",
        len(new_cols),
        half_life,
    )
    return df
