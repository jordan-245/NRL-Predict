"""
Main feature-engineering module for the NRL Match Winner Prediction project.

Each ``build_*`` function computes a specific group of pre-match features
and returns the enriched DataFrame.  The ``build_all_features`` orchestrator
calls them in the correct order and produces the final feature matrix ready
for modelling.

**Critical invariant**: every feature is computed exclusively from data
available *before* the match in question.  No future information ever leaks
into a feature row.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from processing.elo import EloRating
from processing.rolling_stats import (
    compute_exponential_weighted,
    compute_h2h_features,
    compute_rolling_form,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Constants
# ===================================================================

# Known rivalries (unordered pairs of canonical team names)
_RIVALRY_PAIRS: set[frozenset[str]] = {
    frozenset({"South Sydney Rabbitohs", "Sydney Roosters"}),
    frozenset({"Canterbury Bulldogs", "Parramatta Eels"}),
    frozenset({"Brisbane Broncos", "North Queensland Cowboys"}),
    frozenset({"Manly Sea Eagles", "Parramatta Eels"}),
    frozenset({"Cronulla Sharks", "St George Illawarra Dragons"}),
    frozenset({"Melbourne Storm", "Brisbane Broncos"}),
    frozenset({"Newcastle Knights", "Manly Sea Eagles"}),
    frozenset({"Penrith Panthers", "Parramatta Eels"}),
    frozenset({"Canberra Raiders", "New Zealand Warriors"}),
    frozenset({"Wests Tigers", "Parramatta Eels"}),
}

# NRL positional groups by jersey number
_SPINE_POSITIONS = {1, 6, 7, 9}       # Fullback, five-eighth, halfback, hooker
_BACK_POSITIONS = {2, 3, 4, 5}        # Wings and centres
_FORWARD_POSITIONS = {8, 9, 10, 11, 12, 13}  # Props, second-row, lock (9=hooker shared)
_BENCH_POSITIONS = {14, 15, 16, 17}

# Time-slot classification thresholds (local hour)
_TIME_SLOT_MAP = {
    range(0, 12): "morning",
    range(12, 16): "afternoon",
    range(16, 19): "evening",
    range(19, 24): "night",
}


def _classify_time_slot(hour: Optional[int]) -> str:
    """Map an hour (0-23) to a time-slot label."""
    if hour is None or pd.isna(hour):
        return "unknown"
    for rng, label in _TIME_SLOT_MAP.items():
        if int(hour) in rng:
            return label
    return "unknown"


# ===================================================================
# build_team_features
# ===================================================================

def build_team_features(
    matches_df: pd.DataFrame,
    ladders_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build team-level features from ladder standings.

    Features produced:

    * ``home_is_home`` -- always 1 (useful as a flag in team-agnostic models)
    * ``home_ladder_pos``, ``away_ladder_pos`` -- ladder position entering round
    * ``ladder_pos_diff`` -- home_ladder_pos - away_ladder_pos
    * ``home_win_pct``, ``away_win_pct`` -- season win percentages
    * ``home_for_against``, ``away_for_against`` -- for/against ratio
    * ``home_games_behind``, ``away_games_behind`` -- competition points
      behind the ladder leader

    Parameters
    ----------
    matches_df:
        Cleaned matches DataFrame (must have ``season``, ``round``,
        ``home_team``, ``away_team``).
    ladders_df:
        Cleaned ladder DataFrame.  If None, only the ``home_is_home`` flag
        is produced.

    Returns
    -------
    pd.DataFrame
        Copy of *matches_df* with team-feature columns appended.
    """
    df = matches_df.copy()
    df["home_is_home"] = 1

    if ladders_df is None or ladders_df.empty:
        logger.info("build_team_features: no ladder data; returning minimal features.")
        return df

    ladder = ladders_df.copy()

    # Ensure we have the pre-match ladder.  The ladder after round N is used
    # for round N+1.  So for a match in (season=S, round=R), we look up
    # the ladder for (season=S, round=R-1).  For round 1, there is no prior
    # ladder so features will be NaN.
    ladder["round"] = pd.to_numeric(ladder["round"], errors="coerce")

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        pos_col = f"{side}_ladder_pos"
        wpct_col = f"{side}_win_pct"
        fa_col = f"{side}_for_against"
        gb_col = f"{side}_games_behind"

        positions = []
        win_pcts = []
        fa_ratios = []
        games_behind = []

        for _, row in df.iterrows():
            season = row.get("season")
            round_num = row.get("round")
            team = row.get(team_col)

            # Look up ladder from the previous round
            prev_round = None
            if pd.notna(round_num):
                try:
                    prev_round = int(round_num) - 1
                except (ValueError, TypeError):
                    prev_round = None

            if prev_round is None or prev_round < 1 or pd.isna(season) or pd.isna(team):
                positions.append(np.nan)
                win_pcts.append(np.nan)
                fa_ratios.append(np.nan)
                games_behind.append(np.nan)
                continue

            team_ladder = ladder.loc[
                (ladder["season"] == season)
                & (ladder["round"] == prev_round)
                & (ladder["team"] == team)
            ]

            if team_ladder.empty:
                positions.append(np.nan)
                win_pcts.append(np.nan)
                fa_ratios.append(np.nan)
                games_behind.append(np.nan)
            else:
                row_l = team_ladder.iloc[0]
                positions.append(row_l.get("position", np.nan))
                win_pcts.append(row_l.get("win_pct", np.nan))
                fa_ratios.append(row_l.get("for_against_ratio", np.nan))

                # Games behind: difference in competition points from the leader
                round_ladder = ladder.loc[
                    (ladder["season"] == season) & (ladder["round"] == prev_round)
                ]
                if "points" in round_ladder.columns and not round_ladder.empty:
                    max_pts = round_ladder["points"].max()
                    team_pts = row_l.get("points", np.nan)
                    if pd.notna(max_pts) and pd.notna(team_pts):
                        games_behind.append(max_pts - team_pts)
                    else:
                        games_behind.append(np.nan)
                else:
                    games_behind.append(np.nan)

        df[pos_col] = positions
        df[wpct_col] = win_pcts
        df[fa_col] = fa_ratios
        df[gb_col] = games_behind

    # Derived
    if "home_ladder_pos" in df.columns and "away_ladder_pos" in df.columns:
        df["ladder_pos_diff"] = df["home_ladder_pos"] - df["away_ladder_pos"]

    logger.info("build_team_features: added ladder-based team features.")
    return df


# ===================================================================
# build_schedule_features
# ===================================================================

def build_schedule_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Build schedule / fatigue features.

    Features produced (for each side):

    * ``{side}_days_rest`` -- days since the team's previous match
    * ``{side}_is_back_to_back`` -- True if rest <= 5 days
    * ``{side}_had_bye`` -- True if the team did not play in the immediately
      preceding round
    * ``{side}_games_last_14d`` -- number of games in the 14 days before
    * ``{side}_games_last_21d`` -- number of games in the 21 days before

    Parameters
    ----------
    matches_df:
        Cleaned, chronologically sorted matches DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy with schedule columns appended.
    """
    df = matches_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Build a per-team date list for lookups
    team_dates: dict[str, list[pd.Timestamp]] = {}
    for _, row in df.iterrows():
        dt = row.get("date")
        if pd.isna(dt):
            continue
        for team_col in ("home_team", "away_team"):
            team = row.get(team_col)
            if pd.notna(team):
                team_dates.setdefault(team, []).append(dt)

    # Sort each team's date list
    for team in team_dates:
        team_dates[team] = sorted(team_dates[team])

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        days_rest = []
        back_to_back = []
        had_bye = []
        games_14 = []
        games_21 = []

        # Track each team's game counter to find "previous game"
        team_game_counter: dict[str, int] = {}

        for _, row in df.iterrows():
            team = row.get(team_col)
            match_date = row.get("date")

            if pd.isna(team) or pd.isna(match_date):
                days_rest.append(np.nan)
                back_to_back.append(False)
                had_bye.append(False)
                games_14.append(np.nan)
                games_21.append(np.nan)
                continue

            t_dates = team_dates.get(team, [])
            # Current position in the team's chronological list
            idx = team_game_counter.get(team, 0)
            team_game_counter[team] = idx + 1

            # Days since previous game
            if idx > 0 and idx <= len(t_dates):
                prev_date = t_dates[idx - 1]
                rest = (match_date - prev_date).days
                days_rest.append(rest)
                back_to_back.append(rest <= 5)
            else:
                days_rest.append(np.nan)
                back_to_back.append(False)

            # Bye detection: check if team had a round gap
            # Simple heuristic: if rest > 10 days and it's not the start
            # of the season, the team likely had a bye
            if idx > 0 and idx <= len(t_dates):
                rest_val = (match_date - t_dates[idx - 1]).days
                had_bye.append(rest_val >= 11)
            else:
                had_bye.append(False)

            # Games in last N days
            prior_dates = t_dates[:idx]
            g14 = sum(
                1
                for d in prior_dates
                if (match_date - d).days <= 14
            )
            g21 = sum(
                1
                for d in prior_dates
                if (match_date - d).days <= 21
            )
            games_14.append(g14)
            games_21.append(g21)

        df[f"{side}_days_rest"] = days_rest
        df[f"{side}_is_back_to_back"] = back_to_back
        df[f"{side}_had_bye"] = had_bye
        df[f"{side}_games_last_14d"] = games_14
        df[f"{side}_games_last_21d"] = games_21

    logger.info("build_schedule_features: added schedule/fatigue features.")
    return df


# ===================================================================
# build_lineup_features
# ===================================================================

def build_lineup_features(
    matches_df: pd.DataFrame,
    lineups_df: Optional[pd.DataFrame] = None,
    players_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build lineup-based features from player and lineup data.

    Features produced (for each side):

    * ``{side}_total_career_apps`` -- sum of career appearances of starting
      lineup (proxy for experience)
    * ``{side}_lineup_changes`` -- number of player changes from the
      previous round's lineup
    * ``{side}_debutant_count`` -- number of players making their NRL debut
      (career appearances == 0 or 1)
    * ``{side}_spine_same_as_last`` -- whether the halfback (7) and
      fullback (1) are unchanged from the previous game
    * ``{side}_spine_experience`` -- total career appearances of spine
      players (1, 6, 7, 9)
    * ``{side}_forward_experience`` -- total career appearances of forwards
      (8-13)
    * ``{side}_back_experience`` -- total career appearances of backs (2-5)

    Parameters
    ----------
    matches_df:
        Cleaned, chronologically sorted matches DataFrame.
    lineups_df:
        Cleaned lineups DataFrame (player-level rows).
    players_df:
        Cleaned players DataFrame with career statistics.

    Returns
    -------
    pd.DataFrame
        Copy with lineup columns appended.
    """
    df = matches_df.copy()

    if lineups_df is None or lineups_df.empty:
        logger.info("build_lineup_features: no lineup data; skipping.")
        return df

    lineups = lineups_df.copy()

    # Build a player career appearances lookup from players_df
    career_apps: dict[str, int] = {}
    if players_df is not None and not players_df.empty:
        for _, p_row in players_df.iterrows():
            name = p_row.get("player_name", "")
            apps = p_row.get("career_appearances", p_row.get("appearances", 0))
            if pd.notna(name) and pd.notna(apps):
                career_apps[str(name).strip()] = int(apps)

    # Determine match-lineup join key
    if "match_id" in lineups.columns and "match_id" in df.columns:
        match_key = "match_id"
    else:
        match_key = None

    # Build a lookup: (match_identifier, team) -> list of (player, position)
    def _get_match_key_val(row: pd.Series) -> Optional[tuple]:
        if match_key:
            return (row.get(match_key),)
        else:
            return (row.get("season"), row.get("round"), row.get("home_team"), row.get("away_team"))

    lineup_by_match: dict[tuple, dict[str, list[tuple[str, Optional[int]]]]] = {}
    for _, l_row in lineups.iterrows():
        key = _get_match_key_val(l_row)
        team = l_row.get("team", "")
        player = l_row.get("player_name", "")
        position = l_row.get("position")
        if pd.notna(team):
            lineup_by_match.setdefault(key, {}).setdefault(str(team), []).append(
                (str(player) if pd.notna(player) else "", int(position) if pd.notna(position) else None)
            )

    # Track previous lineup per team for change detection
    prev_lineup_by_team: dict[str, set[str]] = {}
    prev_spine_by_team: dict[str, dict[int, str]] = {}

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        total_apps = []
        lineup_changes = []
        debutant_counts = []
        spine_same = []
        spine_exp = []
        forward_exp = []
        back_exp = []

        for _, row in df.iterrows():
            team = row.get(team_col)
            key = _get_match_key_val(row)
            match_lineup = lineup_by_match.get(key, {}).get(str(team), [])

            if not match_lineup:
                total_apps.append(np.nan)
                lineup_changes.append(np.nan)
                debutant_counts.append(np.nan)
                spine_same.append(np.nan)
                spine_exp.append(np.nan)
                forward_exp.append(np.nan)
                back_exp.append(np.nan)
                continue

            # Career appearances
            apps_list = [career_apps.get(p, 0) for p, pos in match_lineup]
            total_apps.append(sum(apps_list))

            # Debutants (0 or 1 career appearance)
            debutant_counts.append(sum(1 for a in apps_list if a <= 1))

            # Positional group experience
            s_exp = sum(
                career_apps.get(p, 0) for p, pos in match_lineup if pos in _SPINE_POSITIONS
            )
            f_exp = sum(
                career_apps.get(p, 0) for p, pos in match_lineup if pos in _FORWARD_POSITIONS
            )
            b_exp = sum(
                career_apps.get(p, 0) for p, pos in match_lineup if pos in _BACK_POSITIONS
            )
            spine_exp.append(s_exp)
            forward_exp.append(f_exp)
            back_exp.append(b_exp)

            # Lineup changes from previous game
            current_players = {p for p, _ in match_lineup if p}
            prev_players = prev_lineup_by_team.get(str(team), set())
            if prev_players:
                changes = len(current_players.symmetric_difference(prev_players)) // 2
                lineup_changes.append(changes)
            else:
                lineup_changes.append(np.nan)

            # Spine continuity: same fullback (1) and halfback (7)?
            current_spine = {pos: p for p, pos in match_lineup if pos in {1, 7} and p}
            prev_spine = prev_spine_by_team.get(str(team), {})
            if prev_spine:
                fb_same = current_spine.get(1) == prev_spine.get(1)
                hb_same = current_spine.get(7) == prev_spine.get(7)
                spine_same.append(1 if (fb_same and hb_same) else 0)
            else:
                spine_same.append(np.nan)

            # Update previous lineup tracking
            prev_lineup_by_team[str(team)] = current_players
            prev_spine_by_team[str(team)] = current_spine

        df[f"{side}_total_career_apps"] = total_apps
        df[f"{side}_lineup_changes"] = lineup_changes
        df[f"{side}_debutant_count"] = debutant_counts
        df[f"{side}_spine_same_as_last"] = spine_same
        df[f"{side}_spine_experience"] = spine_exp
        df[f"{side}_forward_experience"] = forward_exp
        df[f"{side}_back_experience"] = back_exp

    logger.info("build_lineup_features: added lineup features for both sides.")
    return df


# ===================================================================
# build_venue_features
# ===================================================================

def build_venue_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Build venue-related features.

    Features produced:

    * ``is_home_ground`` -- True if the home team has played > 60% of their
      prior home games at this venue (in the same season).
    * ``home_venue_win_rate`` -- home team's historical win rate at this venue
    * ``away_venue_win_rate`` -- away team's historical win rate at this venue
    * ``venue_avg_total_score`` -- historical average combined score at venue

    Parameters
    ----------
    matches_df:
        Cleaned, chronologically sorted matches DataFrame (must include
        ``venue``).

    Returns
    -------
    pd.DataFrame
        Copy with venue columns appended.
    """
    df = matches_df.copy()

    if "venue" not in df.columns:
        logger.info("build_venue_features: no venue column; skipping.")
        return df

    df = df.sort_values("date").reset_index(drop=True)

    # Accumulate venue stats as we iterate (no leakage)
    # (team, venue) -> [win_count, game_count]
    team_venue_record: dict[tuple[str, str], list[int]] = {}
    # venue -> [total_score_sum, game_count]
    venue_scoring: dict[str, list[float]] = {}
    # (team, season) -> {venue: count}  -- for home ground detection
    team_season_venues: dict[tuple[str, int], dict[str, int]] = {}

    is_home_ground = []
    home_venue_wr = []
    away_venue_wr = []
    venue_avg_score = []

    for _, row in df.iterrows():
        venue = row.get("venue", "")
        home = row.get("home_team", "")
        away = row.get("away_team", "")
        season = row.get("season")
        home_score = row.get("home_score")
        away_score = row.get("away_score")

        if not venue or pd.isna(venue) or venue in ("nan", ""):
            is_home_ground.append(np.nan)
            home_venue_wr.append(np.nan)
            away_venue_wr.append(np.nan)
            venue_avg_score.append(np.nan)
            continue

        venue = str(venue).strip()

        # Is this the home team's usual ground?
        if pd.notna(season) and pd.notna(home):
            season_venues = team_season_venues.get((home, int(season)), {})
            total_home_games = sum(season_venues.values())
            venue_games = season_venues.get(venue, 0)
            if total_home_games > 0:
                is_home_ground.append(venue_games / total_home_games > 0.6)
            else:
                is_home_ground.append(np.nan)
        else:
            is_home_ground.append(np.nan)

        # Historical win rate at this venue (PRE-match)
        h_rec = team_venue_record.get((home, venue), [0, 0])
        a_rec = team_venue_record.get((away, venue), [0, 0])
        home_venue_wr.append(h_rec[0] / h_rec[1] if h_rec[1] > 0 else np.nan)
        away_venue_wr.append(a_rec[0] / a_rec[1] if a_rec[1] > 0 else np.nan)

        # Average total score at venue (PRE-match)
        v_scoring = venue_scoring.get(venue, [0.0, 0])
        venue_avg_score.append(
            v_scoring[0] / v_scoring[1] if v_scoring[1] > 0 else np.nan
        )

        # --- Update accumulators AFTER recording pre-match features -------
        if pd.notna(home_score) and pd.notna(away_score):
            total = float(home_score) + float(away_score)
            is_home_win = float(home_score) > float(away_score)
            is_away_win = float(away_score) > float(home_score)

            # Home team at this venue
            rec = team_venue_record.setdefault((home, venue), [0, 0])
            rec[0] += int(is_home_win)
            rec[1] += 1

            # Away team at this venue
            rec = team_venue_record.setdefault((away, venue), [0, 0])
            rec[0] += int(is_away_win)
            rec[1] += 1

            # Venue total scoring
            v = venue_scoring.setdefault(venue, [0.0, 0])
            v[0] += total
            v[1] += 1

            # Home-ground tracking
            if pd.notna(season):
                sv = team_season_venues.setdefault((home, int(season)), {})
                sv[venue] = sv.get(venue, 0) + 1

    df["is_home_ground"] = is_home_ground
    df["home_venue_win_rate"] = home_venue_wr
    df["away_venue_win_rate"] = away_venue_wr
    df["venue_avg_total_score"] = venue_avg_score

    logger.info("build_venue_features: added venue features.")
    return df


# ===================================================================
# build_contextual_features
# ===================================================================

def build_contextual_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Build contextual / calendar features.

    Features produced:

    * ``round_number`` -- numeric round (NaN for finals)
    * ``is_finals`` -- boolean (carried forward if already present)
    * ``day_of_week`` -- 0=Monday ... 6=Sunday
    * ``time_slot`` -- ``"morning"`` / ``"afternoon"`` / ``"evening"`` /
      ``"night"`` / ``"unknown"``
    * ``season_year`` -- season year as int
    * ``rivalry_flag`` -- True if the matchup is a known NRL rivalry

    Parameters
    ----------
    matches_df:
        Cleaned matches DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy with contextual columns appended.
    """
    df = matches_df.copy()

    # Round number
    if "round" in df.columns:
        df["round_number"] = pd.to_numeric(df["round"], errors="coerce")

    # Finals flag (may already be set by clean_matches)
    if "is_finals" not in df.columns and "round" in df.columns:
        round_str = df["round"].astype(str).str.strip().str.lower()
        finals_keywords = {"final", "qualif", "elim", "semi", "prelim", "grand"}
        df["is_finals"] = round_str.apply(
            lambda r: any(kw in r for kw in finals_keywords)
        )

    # Day of week
    if "date" in df.columns:
        df["day_of_week"] = pd.to_datetime(df["date"], errors="coerce").dt.dayofweek

    # Time slot
    if "kickoff" in df.columns:
        kickoff_dt = pd.to_datetime(df["kickoff"], errors="coerce")
        df["time_slot"] = kickoff_dt.dt.hour.apply(_classify_time_slot)
    elif "date" in df.columns:
        # If date includes time component
        dates = pd.to_datetime(df["date"], errors="coerce")
        hours = dates.dt.hour
        # Only classify if the hour is non-zero (i.e. time info present)
        df["time_slot"] = hours.apply(
            lambda h: _classify_time_slot(h) if h != 0 else "unknown"
        )

    # Season year
    if "season" in df.columns:
        df["season_year"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    elif "date" in df.columns:
        df["season_year"] = pd.to_datetime(
            df["date"], errors="coerce"
        ).dt.year.astype("Int64")

    # Rivalry flag
    df["rivalry_flag"] = df.apply(
        lambda r: frozenset({r.get("home_team", ""), r.get("away_team", "")})
        in _RIVALRY_PAIRS,
        axis=1,
    )

    logger.info("build_contextual_features: added contextual features.")
    return df


# ===================================================================
# build_odds_features
# ===================================================================

def build_odds_features(
    matches_df: pd.DataFrame,
    odds_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build odds-derived features.

    Only uses **opening** odds to avoid information leakage from late-market
    movements driven by team news.

    Features produced:

    * ``home_open_prob`` -- implied probability from opening home odds
    * ``away_open_prob`` -- implied probability from opening away odds
    * ``odds_home_favourite`` -- 1 if home is the favourite, else 0
    * ``odds_movement`` -- closing implied prob minus opening implied prob
      for the home team (indicates line movement / information)

    Parameters
    ----------
    matches_df:
        Matches DataFrame (may already have odds columns from linking).
    odds_df:
        Optional separate odds DataFrame.  If provided, it is merged;
        otherwise odds columns already in *matches_df* are used.

    Returns
    -------
    pd.DataFrame
        Copy with odds-feature columns appended.
    """
    df = matches_df.copy()

    # If a separate odds_df is provided, merge it
    if odds_df is not None and not odds_df.empty:
        from processing.data_linking import link_matches_odds

        df = link_matches_odds(df, odds_df)

    # Opening odds implied probabilities
    if "home_open_implied_prob" in df.columns:
        df["home_open_prob"] = df["home_open_implied_prob"]
    elif "home_odds_open" in df.columns:
        df["home_open_prob"] = 1.0 / df["home_odds_open"]
    elif "home_implied_prob_fair" in df.columns:
        df["home_open_prob"] = df["home_implied_prob_fair"]
    elif "home_odds" in df.columns:
        df["home_open_prob"] = 1.0 / df["home_odds"]

    if "away_open_implied_prob" in df.columns:
        df["away_open_prob"] = df["away_open_implied_prob"]
    elif "away_odds_open" in df.columns:
        df["away_open_prob"] = 1.0 / df["away_odds_open"]
    elif "away_implied_prob_fair" in df.columns:
        df["away_open_prob"] = df["away_implied_prob_fair"]
    elif "away_odds" in df.columns:
        df["away_open_prob"] = 1.0 / df["away_odds"]

    # Home favourite flag
    if "home_open_prob" in df.columns and "away_open_prob" in df.columns:
        df["odds_home_favourite"] = (
            df["home_open_prob"] > df["away_open_prob"]
        ).astype(int)

    # Odds movement (closing - opening) for home team
    home_close = None
    if "home_odds_close" in df.columns:
        home_close = 1.0 / df["home_odds_close"]
    elif "home_implied_prob_fair" in df.columns:
        home_close = df["home_implied_prob_fair"]

    home_open = df.get("home_open_prob")
    if home_close is not None and home_open is not None:
        df["odds_movement"] = home_close - home_open
    else:
        df["odds_movement"] = np.nan

    logger.info("build_odds_features: added odds-derived features.")
    return df


# ===================================================================
# build_all_features  (Orchestrator)
# ===================================================================

def build_all_features(
    matches: pd.DataFrame,
    lineups: Optional[pd.DataFrame] = None,
    ladders: Optional[pd.DataFrame] = None,
    players: Optional[pd.DataFrame] = None,
    odds: Optional[pd.DataFrame] = None,
    feature_version: str = "v2",
    elo_kwargs: Optional[dict] = None,
    rolling_windows: Optional[List[int]] = None,
    ewma_half_life: int = 5,
    h2h_lookbacks: Optional[list] = None,
) -> pd.DataFrame:
    """Orchestrate the full feature-engineering pipeline.

    Calls the individual ``build_*`` functions in the correct order and
    returns a single DataFrame with all features attached.

    Feature versions
    ~~~~~~~~~~~~~~~~
    * ``v1`` -- Elo, home/away flag, ladder position, days rest, round
      number, contextual features.
    * ``v2`` -- v1 + rolling form (3/5/8/10), H2H record, EWMA, venue
      features.
    * ``v3`` -- v2 + lineup experience, changes, spine continuity.
    * ``v4`` -- v3 + odds features.

    Parameters
    ----------
    matches:
        Cleaned matches DataFrame (chronologically sorted).
    lineups:
        Cleaned lineups DataFrame.
    ladders:
        Cleaned ladder DataFrame.
    players:
        Cleaned players DataFrame.
    odds:
        Cleaned odds DataFrame.
    feature_version:
        Which feature set to build.  Default ``"v2"``.
    elo_kwargs:
        Optional overrides for the Elo rating system (e.g. ``k_factor``).
    rolling_windows:
        Window sizes for rolling form.  Default ``[3, 5, 8, 10]``.
    ewma_half_life:
        Half-life for exponential weighting.  Default 5.
    h2h_lookbacks:
        Head-to-head lookback specifications.  Default ``[3, 5, "3years"]``.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per match.
    """
    version = feature_version.lower().strip()
    if rolling_windows is None:
        rolling_windows = [3, 5, 8, 10]
    if h2h_lookbacks is None:
        h2h_lookbacks = [3, 5, "3years"]
    if elo_kwargs is None:
        elo_kwargs = {}

    logger.info("build_all_features: building feature set '%s'.", version)
    df = matches.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # v1 features: Elo, team, schedule, context
    # ------------------------------------------------------------------
    # Elo
    elo = EloRating(**elo_kwargs)
    df = elo.backfill(df)
    logger.info("  Elo backfill complete.")

    # Team / ladder features
    df = build_team_features(df, ladders)
    logger.info("  Team features complete.")

    # Schedule features
    df = build_schedule_features(df)
    logger.info("  Schedule features complete.")

    # Contextual features
    df = build_contextual_features(df)
    logger.info("  Contextual features complete.")

    if version == "v1":
        logger.info("build_all_features: v1 done. Shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # v2 features: + rolling form, H2H, EWMA, venue
    # ------------------------------------------------------------------
    df = compute_rolling_form(df, windows=rolling_windows)
    logger.info("  Rolling form complete.")

    df = compute_h2h_features(df, lookbacks=h2h_lookbacks)
    logger.info("  H2H features complete.")

    df = compute_exponential_weighted(df, half_life=ewma_half_life)
    logger.info("  EWMA features complete.")

    df = build_venue_features(df)
    logger.info("  Venue features complete.")

    if version == "v2":
        logger.info("build_all_features: v2 done. Shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # v3 features: + lineup
    # ------------------------------------------------------------------
    df = build_lineup_features(df, lineups, players)
    logger.info("  Lineup features complete.")

    if version == "v3":
        logger.info("build_all_features: v3 done. Shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # v4 features: + odds
    # ------------------------------------------------------------------
    df = build_odds_features(df, odds)
    logger.info("  Odds features complete.")

    logger.info("build_all_features: %s done. Shape: %s", version, df.shape)
    return df
