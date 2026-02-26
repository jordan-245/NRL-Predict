"""
Compute Elo-adjusted player impact scores.

For each player with sufficient data (≥10 starts AND ≥3 absences on a team),
measures the difference in team performance when the player starts vs when
they don't, adjusting for opponent strength via Elo expected win rate.

Impact = avg(actual_result - elo_expected) WITH player
       - avg(actual_result - elo_expected) WITHOUT player

This isolates the player's contribution from schedule strength.

Output: data/processed/player_impact.parquet

Usage:
    python -m processing.player_impact
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DIR

APPEARANCES_PATH = PROCESSED_DIR / "player_appearances.parquet"
MATCHES_PATH = PROCESSED_DIR / "matches.parquet"
OUTPUT_PATH = PROCESSED_DIR / "player_impact.parquet"

# Minimum thresholds
MIN_STARTS = 10
MIN_ABSENCES = 3

# Rolling window: use last N seasons for current ratings
WINDOW_SEASONS = 3

# Spine positions get extra weight in features
SPINE_POSITIONS = {"FB", "HB", "FE", "HK"}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load player appearances and match results with Elo data."""
    if not APPEARANCES_PATH.exists():
        raise FileNotFoundError(
            f"Player appearances not found at {APPEARANCES_PATH}. "
            "Run: python -m processing.build_player_data"
        )

    appearances = pd.read_parquet(APPEARANCES_PATH)
    matches = pd.read_parquet(MATCHES_PATH)
    return appearances, matches


def build_team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """Build a per-team match log with Elo expected values.

    We need to compute Elo ratings for each match to get expected win rates.
    Since matches.parquet may not have Elo columns, we compute them here.
    """
    from processing.elo import EloRating, MovAdjustment

    # Load Elo params
    import json
    params_path = PROJECT_ROOT / "config" / "elo_params.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
    else:
        params = {"k_factor": 15.0, "home_advantage": 34.1,
                  "season_reset_factor": 0.492, "mov_adjustment": "linear"}

    mov_str = params.get("mov_adjustment", "none")
    elo = EloRating(
        k_factor=params["k_factor"],
        home_advantage=params["home_advantage"],
        season_reset_factor=params["season_reset_factor"],
        mov_adjustment=mov_str,
    )

    df = matches.copy()
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df.get("parsed_date", df.get("date")), errors="coerce")
    df = df.sort_values(["year", "date"]).reset_index(drop=True)

    # Compute Elo for each match
    elo_home_list = []
    elo_away_list = []
    elo_exp_list = []

    for _, row in df.iterrows():
        yr = row["year"]
        home = row["home_team"]
        away = row["away_team"]
        hs = row["home_score"]
        as_ = row["away_score"]

        home_elo = elo.get_rating(home)
        away_elo = elo.get_rating(away)
        exp_home = elo.get_expected(home, away)

        elo_home_list.append(home_elo)
        elo_away_list.append(away_elo)
        elo_exp_list.append(exp_home)

        elo.update(home, away, int(hs), int(as_), season=yr)

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["elo_exp_home"] = elo_exp_list

    # Build per-team log
    home_log = pd.DataFrame({
        "year": df["year"],
        "round": df["round"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "match_id": df.apply(
            lambda r: f"{r['year']}_r{r['round']}_{r['home_team']}_v_{r['away_team']}",
            axis=1
        ),
        "home_away": "home",
        "result": np.where(df["home_score"] > df["away_score"], 1.0,
                           np.where(df["home_score"] < df["away_score"], 0.0, 0.5)),
        "margin": df["home_score"] - df["away_score"],
        "elo_expected": df["elo_exp_home"],
    })

    away_log = pd.DataFrame({
        "year": home_log["year"],
        "round": home_log["round"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "match_id": home_log["match_id"],
        "home_away": "away",
        "result": 1.0 - home_log["result"],
        "margin": -home_log["margin"],
        "elo_expected": 1.0 - df["elo_exp_home"],
    })

    team_log = pd.concat([home_log, away_log], ignore_index=True)
    team_log = team_log.sort_values(["year", "round"]).reset_index(drop=True)

    return team_log


def compute_impact_scores(
    appearances: pd.DataFrame,
    team_log: pd.DataFrame,
    window_seasons: int = WINDOW_SEASONS,
) -> pd.DataFrame:
    """Compute Elo-adjusted impact scores for each player.

    For each (player, team, season_window):
    1. Find all team matches in the window
    2. Split into: player started vs player didn't start
    3. Compute residual = actual_result - elo_expected for each group
    4. Impact = mean_residual_with - mean_residual_without
    """
    print("\n  Computing player impact scores...")

    # Get unique years
    all_years = sorted(appearances["year"].unique())
    if not all_years:
        return pd.DataFrame()

    # Build a set of (match_id, team, player_id) for starters
    starters = appearances[appearances["is_starter"]].copy()
    starter_set = set(
        zip(starters["match_id"], starters["team"], starters["player_id"])
    )

    # Build player→team→years mapping
    player_teams = (
        starters.groupby(["player_id", "team"])["year"]
        .agg(["min", "max", "count"])
        .reset_index()
    )

    # For each player-team combo, compute impact over rolling windows
    results = []
    max_year = max(all_years)

    # Get all unique players who started enough games
    for _, pt_row in player_teams.iterrows():
        player_id = pt_row["player_id"]
        team = pt_row["team"]

        # Use the most recent window
        window_end = max_year
        window_start = window_end - window_seasons + 1

        # Get team matches in window
        team_matches = team_log[
            (team_log["team"] == team) &
            (team_log["year"] >= window_start) &
            (team_log["year"] <= window_end)
        ].copy()

        if len(team_matches) < MIN_STARTS + MIN_ABSENCES:
            continue

        # Classify each match: player started or not
        team_matches["player_started"] = team_matches["match_id"].apply(
            lambda mid: (mid, team, player_id) in starter_set
        )

        with_player = team_matches[team_matches["player_started"]]
        without_player = team_matches[~team_matches["player_started"]]

        games_started = len(with_player)
        games_missed = len(without_player)

        if games_started < MIN_STARTS or games_missed < MIN_ABSENCES:
            continue

        # Elo-adjusted residuals
        residual_with = (with_player["result"] - with_player["elo_expected"]).mean()
        residual_without = (without_player["result"] - without_player["elo_expected"]).mean()

        elo_adj_impact = residual_with - residual_without

        # Margin impact
        margin_with = with_player["margin"].mean()
        margin_without = without_player["margin"].mean()
        margin_impact = margin_with - margin_without

        # Confidence score: based on sample size
        total_games = games_started + games_missed
        min_group = min(games_started, games_missed)
        confidence = min(1.0, min_group / 15.0) * min(1.0, total_games / 40.0)

        # Get player metadata from appearances
        player_apps = starters[
            (starters["player_id"] == player_id) &
            (starters["team"] == team) &
            (starters["year"] >= window_start)
        ]

        if player_apps.empty:
            continue

        # Most common position
        position = player_apps["position"].mode().iloc[0] if not player_apps["position"].mode().empty else "UNK"
        player_name = player_apps["player_name"].mode().iloc[0] if not player_apps["player_name"].mode().empty else player_id

        results.append({
            "player_id": player_id,
            "player_name": player_name,
            "team": team,
            "position": position,
            "is_spine": position in SPINE_POSITIONS,
            "season_window": f"{window_start}-{window_end}",
            "games_started": games_started,
            "games_missed": games_missed,
            "elo_adj_impact": round(elo_adj_impact, 4),
            "margin_impact": round(margin_impact, 2),
            "confidence": round(confidence, 3),
            "weighted_impact": round(elo_adj_impact * confidence, 4),
            "win_rate_with": round(with_player["result"].mean(), 3),
            "win_rate_without": round(without_player["result"].mean(), 3),
        })

    impact_df = pd.DataFrame(results)

    if not impact_df.empty:
        impact_df = impact_df.sort_values("weighted_impact", ascending=False).reset_index(drop=True)
        print(f"  Computed impact scores for {len(impact_df)} player-team combos")
        print(f"  Spine players: {impact_df['is_spine'].sum()}")

        # Print top/bottom 10
        print(f"\n  Top 10 positive impact:")
        for _, row in impact_df.head(10).iterrows():
            print(f"    {row['player_name']:<20s} {row['team']:<30s} "
                  f"{row['position']:>3s}  impact={row['elo_adj_impact']:+.3f}  "
                  f"conf={row['confidence']:.2f}  "
                  f"({row['games_started']}G/{row['games_missed']}M)")

        print(f"\n  Top 10 negative impact (team worse when they play):")
        for _, row in impact_df.tail(10).iterrows():
            print(f"    {row['player_name']:<20s} {row['team']:<30s} "
                  f"{row['position']:>3s}  impact={row['elo_adj_impact']:+.3f}  "
                  f"conf={row['confidence']:.2f}  "
                  f"({row['games_started']}G/{row['games_missed']}M)")
    else:
        print("  WARNING: No players met the minimum thresholds!")

    return impact_df


def get_player_impact(
    team: str,
    player_name: str | None = None,
    player_id: str | None = None,
    impact_df: pd.DataFrame | None = None,
) -> float:
    """Look up a single player's impact score.

    Returns 0.0 if not found (unknown player = neutral impact).
    """
    if impact_df is None:
        if OUTPUT_PATH.exists():
            impact_df = pd.read_parquet(OUTPUT_PATH)
        else:
            return 0.0

    mask = impact_df["team"] == team
    if player_id:
        mask = mask & (impact_df["player_id"] == player_id)
    elif player_name:
        # Try exact match first, then surname match
        name_mask = impact_df["player_name"].str.lower() == player_name.lower()
        if not name_mask.any():
            # Try surname-only match (RLP gives surnames)
            surname = player_name.split()[-1] if player_name else ""
            name_mask = impact_df["player_name"].str.lower().str.endswith(surname.lower())
        mask = mask & name_mask

    matches = impact_df[mask]
    if matches.empty:
        return 0.0

    return float(matches.iloc[0]["weighted_impact"])


def main():
    t_start = time.time()

    print("=" * 60)
    print("  Computing Player Impact Scores")
    print("=" * 60)

    appearances, matches = load_data()
    print(f"\n  Appearances: {len(appearances)} rows")
    print(f"  Matches: {len(matches)} rows")

    team_log = build_team_match_log(matches)
    print(f"  Team match log: {len(team_log)} rows")

    impact_df = compute_impact_scores(appearances, team_log)

    if not impact_df.empty:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        impact_df.to_parquet(OUTPUT_PATH, index=False)
        print(f"\n  Saved to {OUTPUT_PATH}")
        print(f"  Shape: {impact_df.shape}")

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
