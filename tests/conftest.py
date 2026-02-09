"""
Shared pytest fixtures for NRL Match Winner Prediction test suite.

Provides synthetic DataFrames that mirror the structure of real NRL data
at a small scale (~20 rows) to enable fast, deterministic unit tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Teams used across synthetic data
# ---------------------------------------------------------------------------

_TEAMS = [
    "Melbourne Storm",
    "Penrith Panthers",
    "Sydney Roosters",
    "South Sydney Rabbitohs",
    "Canterbury Bulldogs",
    "Cronulla Sharks",
    "Brisbane Broncos",
    "North Queensland Cowboys",
    "Parramatta Eels",
    "Manly Sea Eagles",
]


# ---------------------------------------------------------------------------
# sample_matches_df  (~20 synthetic matches)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_matches_df() -> pd.DataFrame:
    """Return a small synthetic matches DataFrame with 20 rows spanning two seasons."""
    np.random.seed(42)
    n = 20
    dates_2023 = pd.date_range("2023-03-02", periods=10, freq="7D")
    dates_2024 = pd.date_range("2024-03-01", periods=10, freq="7D")
    dates = list(dates_2023) + list(dates_2024)

    home_teams = []
    away_teams = []
    for i in range(n):
        h_idx = i % len(_TEAMS)
        a_idx = (i + 3) % len(_TEAMS)
        home_teams.append(_TEAMS[h_idx])
        away_teams.append(_TEAMS[a_idx])

    home_scores = np.random.randint(6, 40, size=n)
    away_scores = np.random.randint(6, 40, size=n)

    # Ensure at least one draw for edge-case testing
    home_scores[5] = 18
    away_scores[5] = 18

    seasons = [2023] * 10 + [2024] * 10
    rounds = list(range(1, 11)) + list(range(1, 11))

    df = pd.DataFrame({
        "date": dates,
        "season": seasons,
        "round": rounds,
        "home_team": home_teams,
        "away_team": away_teams,
        "home_score": home_scores,
        "away_score": away_scores,
        "venue": [
            "AAMI Park", "BlueBet Stadium", "Allianz Stadium",
            "Accor Stadium", "Belmore Sports Ground", "PointsBet Stadium",
            "Suncorp Stadium", "Qld Country Bank Stadium", "CommBank Stadium",
            "4 Pines Park",
        ] * 2,
    })

    return df


# ---------------------------------------------------------------------------
# sample_lineups_df
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_lineups_df(sample_matches_df: pd.DataFrame) -> pd.DataFrame:
    """Return synthetic lineup data (17 players per team per match)."""
    rows = []
    player_counter = 0
    for _, match in sample_matches_df.iterrows():
        for team_col in ("home_team", "away_team"):
            team = match[team_col]
            for pos in range(1, 18):
                player_counter += 1
                rows.append({
                    "season": match["season"],
                    "round": match["round"],
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "team": team,
                    "player_name": f"Player_{player_counter}",
                    "position": pos,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sample_odds_df
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_odds_df(sample_matches_df: pd.DataFrame) -> pd.DataFrame:
    """Return synthetic odds data aligned with sample_matches_df."""
    np.random.seed(99)
    n = len(sample_matches_df)
    home_odds = np.round(np.random.uniform(1.30, 3.50, size=n), 2)
    away_odds = np.round(np.random.uniform(1.30, 3.50, size=n), 2)

    df = pd.DataFrame({
        "date": sample_matches_df["date"].values,
        "home_team": sample_matches_df["home_team"].values,
        "away_team": sample_matches_df["away_team"].values,
        "home_odds": home_odds,
        "away_odds": away_odds,
        "home_score": sample_matches_df["home_score"].values,
        "away_score": sample_matches_df["away_score"].values,
    })
    return df


# ---------------------------------------------------------------------------
# sample_ladder_df
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_ladder_df() -> pd.DataFrame:
    """Return synthetic round-by-round ladder data for 10 teams across 2 seasons."""
    rows = []
    for season in (2023, 2024):
        for rnd in range(1, 11):
            for pos, team in enumerate(_TEAMS, start=1):
                played = rnd
                won = max(0, rnd - pos + 5)
                drawn = 0
                lost = played - won - drawn
                pf = won * 24 + lost * 10 + np.random.randint(-5, 5)
                pa = lost * 24 + won * 10 + np.random.randint(-5, 5)
                points = won * 2 + drawn
                rows.append({
                    "season": season,
                    "round": rnd,
                    "position": pos,
                    "team": team,
                    "played": played,
                    "won": won,
                    "drawn": drawn,
                    "lost": lost,
                    "points_for": int(pf),
                    "points_against": int(pa),
                    "points": points,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sample_features_df  (small feature matrix with target)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_features_df() -> pd.DataFrame:
    """Return a small synthetic feature matrix with a binary target column."""
    np.random.seed(7)
    n = 50
    df = pd.DataFrame({
        "year": [2020] * 15 + [2021] * 15 + [2022] * 20,
        "round": list(range(1, 16)) + list(range(1, 16)) + list(range(1, 21)),
        "home_elo": np.random.normal(1500, 80, n),
        "away_elo": np.random.normal(1500, 80, n),
        "home_win_rate_5": np.random.uniform(0.2, 0.9, n),
        "away_win_rate_5": np.random.uniform(0.2, 0.9, n),
        "home_ladder_pos": np.random.randint(1, 17, n),
        "away_ladder_pos": np.random.randint(1, 17, n),
        "home_days_rest": np.random.randint(5, 14, n),
        "away_days_rest": np.random.randint(5, 14, n),
    })
    # Target: biased toward home win when home_elo > away_elo
    prob = 1.0 / (1.0 + np.exp(-0.005 * (df["home_elo"] - df["away_elo"])))
    df["home_win"] = (np.random.uniform(0, 1, n) < prob).astype(int)
    return df


# ---------------------------------------------------------------------------
# Convenience: feature columns list
# ---------------------------------------------------------------------------

@pytest.fixture()
def feature_columns() -> list[str]:
    """Return the list of feature column names in sample_features_df."""
    return [
        "home_elo",
        "away_elo",
        "home_win_rate_5",
        "away_win_rate_5",
        "home_ladder_pos",
        "away_ladder_pos",
        "home_days_rest",
        "away_days_rest",
    ]
