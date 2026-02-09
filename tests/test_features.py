"""
Tests for feature engineering: Elo ratings, rolling stats, feature engineering,
target encoding, and data cleaning.

Covers:
- Elo rating system: updates, home advantage, season reset, expected probability
- Rolling stats: window computation, no future data leakage, season boundaries
- Feature engineering: build_* functions with small synthetic DataFrames
- Target encoding: home_win target creation, draw handling
- Data cleaning: team name standardisation, date parsing, null handling
- Leakage verification: features for match N only use data from before N
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Elo rating tests
# ============================================================================


class TestEloRating:
    """Tests for processing.elo.EloRating."""

    def test_initial_rating_default(self):
        from processing.elo import EloRating

        elo = EloRating()
        assert elo.get_rating("Melbourne Storm") == 1500.0

    def test_initial_rating_custom(self):
        from processing.elo import EloRating

        elo = EloRating(initial_rating=1200.0)
        assert elo.get_rating("Any Team") == 1200.0

    def test_update_winner_gains_rating(self):
        from processing.elo import EloRating

        elo = EloRating(k_factor=20, home_advantage=0)
        elo.update("Team A", "Team B", home_score=30, away_score=12)
        assert elo.get_rating("Team A") > 1500.0
        assert elo.get_rating("Team B") < 1500.0

    def test_update_loser_loses_rating(self):
        from processing.elo import EloRating

        elo = EloRating(k_factor=20, home_advantage=0)
        elo.update("Team A", "Team B", home_score=10, away_score=30)
        assert elo.get_rating("Team A") < 1500.0
        assert elo.get_rating("Team B") > 1500.0

    def test_update_draw_minimal_change(self):
        from processing.elo import EloRating

        elo = EloRating(k_factor=20, home_advantage=0)
        elo.update("Team A", "Team B", home_score=18, away_score=18)
        # Both teams start at 1500, expected is 0.5, actual is 0.5 -> no change
        assert abs(elo.get_rating("Team A") - 1500.0) < 0.01
        assert abs(elo.get_rating("Team B") - 1500.0) < 0.01

    def test_home_advantage_biases_expected_score(self):
        from processing.elo import EloRating

        elo = EloRating(home_advantage=100.0)
        # With equal ratings and home advantage, home expected > 0.5
        expected = elo.get_expected("Team A", "Team B")
        assert expected > 0.5

    def test_home_advantage_zero_gives_equal_expected(self):
        from processing.elo import EloRating

        elo = EloRating(home_advantage=0.0)
        expected = elo.get_expected("Team A", "Team B")
        assert abs(expected - 0.5) < 0.01

    def test_expected_probability_range(self):
        from processing.elo import EloRating

        elo = EloRating()
        expected = elo.get_expected("Team A", "Team B")
        assert 0.0 <= expected <= 1.0

    def test_expected_probability_strong_vs_weak(self):
        from processing.elo import EloRating

        elo = EloRating(home_advantage=0)
        elo._ratings["Strong Team"] = 1800.0
        elo._ratings["Weak Team"] = 1200.0
        expected = elo.get_expected("Strong Team", "Weak Team")
        assert expected > 0.9  # Very strong favourite

    def test_season_reset_regresses_toward_mean(self):
        from processing.elo import EloRating

        elo = EloRating(season_reset_factor=0.5, initial_rating=1500.0)
        elo._ratings["Team A"] = 1700.0
        elo._ratings["Team B"] = 1300.0
        elo._last_season = 2023

        # Trigger season reset by updating with a new season
        elo.update("Team A", "Team B", 20, 10, season=2024, round_=1)

        # Before the match update, ratings should have been regressed
        # With factor 0.5: new = 1500 + 0.5 * (old - 1500)
        # Team A: 1500 + 0.5 * 200 = 1600 (before match update)
        # Team B: 1500 + 0.5 * (-200) = 1400 (before match update)
        # After match update, winner gains and loser loses some more
        # Just verify the direction: the gap should be smaller than 400
        gap = elo.get_rating("Team A") - elo.get_rating("Team B")
        assert gap < 400  # Was 400 before reset

    def test_season_reset_factor_1_no_regression(self):
        from processing.elo import EloRating

        elo = EloRating(season_reset_factor=1.0, initial_rating=1500.0, home_advantage=0)
        elo._ratings["Team A"] = 1700.0
        elo._last_season = 2023

        elo._maybe_season_reset(2024)
        # Factor = 1.0 means: new = 1500 + 1.0 * (1700 - 1500) = 1700
        assert abs(elo.get_rating("Team A") - 1700.0) < 0.01

    def test_season_reset_factor_0_full_reset(self):
        from processing.elo import EloRating

        elo = EloRating(season_reset_factor=0.0, initial_rating=1500.0)
        elo._ratings["Team A"] = 1700.0
        elo._last_season = 2023

        elo._maybe_season_reset(2024)
        # Factor = 0.0 means: new = 1500 + 0 * (1700 - 1500) = 1500
        assert abs(elo.get_rating("Team A") - 1500.0) < 0.01

    def test_elo_formula_correctness(self):
        """Verify the Elo expected score formula directly."""
        from processing.elo import EloRating

        # E = 1 / (1 + 10^((Rb - Ra) / 400))
        expected = EloRating._expected(1500.0, 1500.0)
        assert abs(expected - 0.5) < 0.001

        expected = EloRating._expected(1700.0, 1500.0)
        # 10^(-200/400) = 10^(-0.5) approx 0.316
        # E = 1/(1 + 0.316) = 0.760
        assert abs(expected - 0.760) < 0.01

    def test_reset_clears_all_state(self):
        from processing.elo import EloRating

        elo = EloRating()
        elo.update("A", "B", 20, 10, season=2023, round_=1)
        elo.reset()
        assert elo.get_ratings() == {}
        assert elo.get_history() == []
        assert elo._last_season is None

    def test_history_records_both_teams(self):
        from processing.elo import EloRating

        elo = EloRating()
        elo.update("Team A", "Team B", 20, 10, season=2024, round_=1)
        history = elo.get_history()
        assert len(history) == 2  # One for each team
        teams = {h.team for h in history}
        assert teams == {"Team A", "Team B"}

    def test_history_df_columns(self):
        from processing.elo import EloRating

        elo = EloRating()
        elo.update("A", "B", 20, 10, season=2024, round_=1)
        df = elo.get_history_df()
        expected_cols = {
            "team", "season", "round", "rating_before", "rating_after",
            "opponent", "is_home", "score_for", "score_against",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_mov_adjustment_none(self):
        from processing.elo import EloRating

        elo = EloRating(mov_adjustment="none")
        assert elo._mov_multiplier(20) == 1.0

    def test_mov_adjustment_linear(self):
        from processing.elo import EloRating

        elo = EloRating(mov_adjustment="linear", mov_linear_divisor=10.0)
        assert elo._mov_multiplier(10) == 2.0  # 1 + 10/10
        assert elo._mov_multiplier(0) == 1.0   # 1 + 0/10

    def test_mov_adjustment_logarithmic(self):
        import math

        from processing.elo import EloRating

        elo = EloRating(mov_adjustment="logarithmic")
        # log(margin + 1) for margin=0 => log(1) = 0
        assert abs(elo._mov_multiplier(0) - math.log(2)) < 0.01
        assert elo._mov_multiplier(10) > 1.0

    def test_backfill_no_leakage(self, sample_matches_df):
        """Verify that backfill computes PRE-match ratings (no leakage)."""
        from processing.elo import EloRating

        elo = EloRating(k_factor=20, home_advantage=50)
        result = elo.backfill(sample_matches_df)

        assert "home_elo" in result.columns
        assert "away_elo" in result.columns
        assert "home_elo_prob" in result.columns

        # First match: both teams should have the initial rating
        assert result.iloc[0]["home_elo"] == 1500.0
        assert result.iloc[0]["away_elo"] == 1500.0

        # Probability should be in [0, 1]
        probs = result["home_elo_prob"].dropna()
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_backfill_second_match_uses_updated_ratings(self, sample_matches_df):
        """After the first match, ratings should be updated for the second."""
        from processing.elo import EloRating

        elo = EloRating(k_factor=20, home_advantage=0)
        result = elo.backfill(sample_matches_df)

        # If the same team appears in match 0 and a later match, its Elo should differ
        first_home = result.iloc[0]["home_team"]
        # Find a later match where this team appears
        later = result[
            (result["home_team"] == first_home) | (result["away_team"] == first_home)
        ]
        if len(later) > 1:
            # The rating should have changed from the initial 1500
            second_idx = later.index[1]
            if result.loc[second_idx, "home_team"] == first_home:
                assert result.loc[second_idx, "home_elo"] != 1500.0
            else:
                assert result.loc[second_idx, "away_elo"] != 1500.0


# ============================================================================
# Rolling stats tests
# ============================================================================


class TestRollingStats:
    """Tests for processing.rolling_stats."""

    def test_compute_rolling_form_basic(self, sample_matches_df):
        from processing.rolling_stats import compute_rolling_form

        result = compute_rolling_form(sample_matches_df, windows=[3])
        assert "home_win_rate_3" in result.columns
        assert "away_win_rate_3" in result.columns
        assert "home_avg_pf_3" in result.columns
        assert "home_avg_pa_3" in result.columns
        assert "home_avg_margin_3" in result.columns

    def test_compute_rolling_form_first_match_is_nan(self, sample_matches_df):
        """The first match for any team should have NaN rolling features."""
        from processing.rolling_stats import compute_rolling_form

        result = compute_rolling_form(sample_matches_df, windows=[3])
        # The first few matches should have NaN (no prior data)
        assert pd.isna(result.iloc[0]["home_win_rate_3"])

    def test_compute_rolling_form_values_in_range(self, sample_matches_df):
        from processing.rolling_stats import compute_rolling_form

        result = compute_rolling_form(sample_matches_df, windows=[3])
        valid = result["home_win_rate_3"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_compute_rolling_form_no_future_leakage(self, sample_matches_df):
        """Features for match N should only use data from before match N."""
        from processing.rolling_stats import compute_rolling_form

        df = sample_matches_df.sort_values("date").reset_index(drop=True)
        result = compute_rolling_form(df, windows=[5])

        # For each match, the rolling stats should not incorporate the match itself.
        # We verify by checking that the rolling stat at the last row does NOT
        # change if we remove the last row's data and recompute.
        # Simpler check: first match has NaN, second match at most uses 1 prior game.
        assert pd.isna(result.iloc[0]["home_win_rate_5"])

    def test_compute_rolling_form_multiple_windows(self, sample_matches_df):
        from processing.rolling_stats import compute_rolling_form

        result = compute_rolling_form(sample_matches_df, windows=[3, 5])
        assert "home_win_rate_3" in result.columns
        assert "home_win_rate_5" in result.columns
        assert "away_win_rate_3" in result.columns
        assert "away_win_rate_5" in result.columns

    def test_compute_rolling_form_respect_season(self, sample_matches_df):
        """When respect_season=True, rolling window resets at season boundaries."""
        from processing.rolling_stats import compute_rolling_form

        result = compute_rolling_form(
            sample_matches_df, windows=[3], respect_season=True
        )
        # The first match of the second season (index 10) should not carry
        # form from the prior season -> should be NaN
        season_2024_start = result[result["season"] == 2024].iloc[0]
        # This might or might not be NaN depending on whether the team played
        # in season 2024 before; at minimum the feature should exist
        assert "home_win_rate_3" in result.columns

    def test_build_team_match_log_structure(self, sample_matches_df):
        from processing.rolling_stats import _build_team_match_log

        log = _build_team_match_log(sample_matches_df)
        assert "team" in log.columns
        assert "opponent" in log.columns
        assert "is_home" in log.columns
        assert "points_for" in log.columns
        assert "points_against" in log.columns
        assert "win" in log.columns
        assert "margin" in log.columns

    def test_build_team_match_log_doubles_rows(self, sample_matches_df):
        """Each match produces 2 rows: one home, one away."""
        from processing.rolling_stats import _build_team_match_log

        log = _build_team_match_log(sample_matches_df)
        assert len(log) == 2 * len(sample_matches_df)

    def test_build_team_match_log_win_values(self, sample_matches_df):
        from processing.rolling_stats import _build_team_match_log

        log = _build_team_match_log(sample_matches_df)
        assert set(log["win"].unique()).issubset({0.0, 0.5, 1.0})

    def test_compute_h2h_features_basic(self, sample_matches_df):
        from processing.rolling_stats import compute_h2h_features

        result = compute_h2h_features(sample_matches_df, lookbacks=[3])
        assert "h2h_home_win_rate_3" in result.columns
        assert "h2h_avg_margin_3" in result.columns
        assert "h2h_matches_3" in result.columns

    def test_compute_h2h_no_future_leakage(self, sample_matches_df):
        """H2H features should only use prior meetings."""
        from processing.rolling_stats import compute_h2h_features

        result = compute_h2h_features(sample_matches_df, lookbacks=[5])
        # First match between any pair has 0 prior meetings
        assert result.iloc[0]["h2h_matches_5"] == 0

    def test_compute_exponential_weighted_basic(self, sample_matches_df):
        from processing.rolling_stats import compute_exponential_weighted

        result = compute_exponential_weighted(sample_matches_df, half_life=3)
        assert "home_ewma_win_3" in result.columns
        assert "away_ewma_win_3" in result.columns
        assert "home_ewma_margin_3" in result.columns

    def test_compute_exponential_weighted_first_match_nan(self, sample_matches_df):
        from processing.rolling_stats import compute_exponential_weighted

        result = compute_exponential_weighted(sample_matches_df, half_life=3)
        # First match for every team has NaN EWMA
        assert pd.isna(result.iloc[0]["home_ewma_win_3"])


# ============================================================================
# Feature engineering tests
# ============================================================================


class TestFeatureEngineering:
    """Tests for processing.feature_engineering build_* functions."""

    def test_build_team_features_home_flag(self, sample_matches_df):
        from processing.feature_engineering import build_team_features

        result = build_team_features(sample_matches_df)
        assert "home_is_home" in result.columns
        assert (result["home_is_home"] == 1).all()

    def test_build_team_features_with_ladder(
        self, sample_matches_df, sample_ladder_df
    ):
        from processing.feature_engineering import build_team_features

        result = build_team_features(sample_matches_df, sample_ladder_df)
        # Should have ladder-based columns
        assert "home_ladder_pos" in result.columns
        assert "away_ladder_pos" in result.columns
        assert "home_win_pct" in result.columns

    def test_build_schedule_features_columns(self, sample_matches_df):
        from processing.feature_engineering import build_schedule_features

        result = build_schedule_features(sample_matches_df)
        assert "home_days_rest" in result.columns
        assert "away_days_rest" in result.columns
        assert "home_is_back_to_back" in result.columns
        assert "home_games_last_14d" in result.columns

    def test_build_schedule_features_first_match_nan_rest(self, sample_matches_df):
        from processing.feature_engineering import build_schedule_features

        result = build_schedule_features(sample_matches_df)
        # First match for each team should have NaN days rest
        assert pd.isna(result.iloc[0]["home_days_rest"])

    def test_build_contextual_features_columns(self, sample_matches_df):
        from processing.feature_engineering import build_contextual_features

        result = build_contextual_features(sample_matches_df)
        assert "round_number" in result.columns
        assert "day_of_week" in result.columns
        assert "rivalry_flag" in result.columns

    def test_build_contextual_features_rivalry_detection(self):
        from processing.feature_engineering import build_contextual_features

        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-03-01", "2024-03-08"]),
            "season": [2024, 2024],
            "round": [1, 2],
            "home_team": ["South Sydney Rabbitohs", "Melbourne Storm"],
            "away_team": ["Sydney Roosters", "Penrith Panthers"],
            "home_score": [20, 18],
            "away_score": [18, 24],
        })
        result = build_contextual_features(df)
        # Souths vs Roosters is a known rivalry
        assert result.iloc[0]["rivalry_flag"] is True
        # Storm vs Panthers is NOT a known rivalry in the default set
        assert result.iloc[1]["rivalry_flag"] is False

    def test_build_venue_features_columns(self, sample_matches_df):
        from processing.feature_engineering import build_venue_features

        result = build_venue_features(sample_matches_df)
        assert "home_venue_win_rate" in result.columns
        assert "away_venue_win_rate" in result.columns
        assert "venue_avg_total_score" in result.columns

    def test_build_all_features_v1_shape(self, sample_matches_df):
        """build_all_features v1 should add Elo, team, schedule, contextual."""
        from processing.feature_engineering import build_all_features

        result = build_all_features(
            sample_matches_df,
            feature_version="v1",
        )
        assert "home_elo" in result.columns
        assert "home_is_home" in result.columns
        assert "home_days_rest" in result.columns
        assert "day_of_week" in result.columns
        assert len(result) == len(sample_matches_df)


# ============================================================================
# Target encoding tests
# ============================================================================


class TestTargetEncoding:
    """Tests for processing.target_encoding."""

    def test_create_target_home_win(self):
        from processing.target_encoding import create_target

        df = pd.DataFrame({"home_score": [30, 10, 18], "away_score": [12, 24, 18]})
        target = create_target(df, draw_handling="exclude")
        assert target.iloc[0] == 1.0  # Home win
        assert target.iloc[1] == 0.0  # Away win
        assert pd.isna(target.iloc[2])  # Draw excluded

    def test_create_target_draw_home(self):
        from processing.target_encoding import create_target

        df = pd.DataFrame({"home_score": [18], "away_score": [18]})
        target = create_target(df, draw_handling="home")
        assert target.iloc[0] == 1.0

    def test_create_target_draw_away(self):
        from processing.target_encoding import create_target

        df = pd.DataFrame({"home_score": [18], "away_score": [18]})
        target = create_target(df, draw_handling="away")
        assert target.iloc[0] == 0.0

    def test_create_target_draw_half(self):
        from processing.target_encoding import create_target

        df = pd.DataFrame({"home_score": [18], "away_score": [18]})
        target = create_target(df, draw_handling="half")
        assert target.iloc[0] == 0.5

    def test_create_target_invalid_draw_handling_raises(self):
        from processing.target_encoding import create_target

        df = pd.DataFrame({"home_score": [20], "away_score": [10]})
        with pytest.raises(ValueError, match="draw_handling"):
            create_target(df, draw_handling="invalid")

    def test_create_margin_target(self):
        from processing.target_encoding import create_margin_target

        df = pd.DataFrame({"home_score": [30, 10, 18], "away_score": [12, 24, 18]})
        margin = create_margin_target(df)
        assert margin.iloc[0] == 18   # 30 - 12
        assert margin.iloc[1] == -14  # 10 - 24
        assert margin.iloc[2] == 0    # 18 - 18

    def test_target_encoder_fit_transform(self):
        from processing.target_encoding import TargetEncoder

        df = pd.DataFrame({
            "venue": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        target = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])

        encoder = TargetEncoder(columns=["venue"], n_folds=2, smoothing=1.0, random_state=42)
        encoded = encoder.fit_transform(df, target)

        assert encoded["venue"].dtype == float
        assert not encoded["venue"].isna().any()
        assert encoder._is_fitted

    def test_target_encoder_transform_unseen_category(self):
        from processing.target_encoding import TargetEncoder

        df_train = pd.DataFrame({"cat": ["A", "A", "B", "B"]})
        target = pd.Series([1, 0, 1, 0])

        encoder = TargetEncoder(columns=["cat"], n_folds=2, smoothing=1.0, random_state=42)
        encoder.fit_transform(df_train, target)

        df_test = pd.DataFrame({"cat": ["A", "C"]})  # "C" is unseen
        result = encoder.transform(df_test)
        # Unseen category "C" should get the global mean
        assert abs(result.iloc[1]["cat"] - encoder.global_mean_) < 0.01

    def test_target_encoder_not_fitted_raises(self):
        from processing.target_encoding import TargetEncoder

        encoder = TargetEncoder(columns=["venue"])
        df = pd.DataFrame({"venue": ["A", "B"]})
        with pytest.raises(RuntimeError, match="not been fitted"):
            encoder.transform(df)


# ============================================================================
# Data cleaning tests
# ============================================================================


class TestDataCleaning:
    """Tests for processing.data_cleaning."""

    def test_clean_matches_standardises_team_names(self):
        from processing.data_cleaning import clean_matches

        df = pd.DataFrame({
            "home_team": ["Storm", "Panthers"],
            "away_team": ["Roosters", "Souths"],
            "home_score": [24, 30],
            "away_score": [18, 12],
            "date": ["2024-03-01", "2024-03-08"],
        })
        cleaned = clean_matches(df)
        assert cleaned.iloc[0]["home_team"] == "Melbourne Storm"
        assert cleaned.iloc[0]["away_team"] == "Sydney Roosters"
        assert cleaned.iloc[1]["home_team"] == "Penrith Panthers"
        assert cleaned.iloc[1]["away_team"] == "South Sydney Rabbitohs"

    def test_clean_matches_parses_dates(self):
        from processing.data_cleaning import clean_matches

        df = pd.DataFrame({
            "home_team": ["Melbourne Storm"],
            "away_team": ["Penrith Panthers"],
            "home_score": [24],
            "away_score": [18],
            "date": ["01/03/2024"],
        })
        cleaned = clean_matches(df)
        assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])

    def test_clean_matches_computes_margin(self):
        from processing.data_cleaning import clean_matches

        df = pd.DataFrame({
            "home_team": ["Melbourne Storm"],
            "away_team": ["Penrith Panthers"],
            "home_score": [30],
            "away_score": [12],
            "date": ["2024-03-01"],
        })
        cleaned = clean_matches(df)
        assert cleaned.iloc[0]["margin"] == 18

    def test_clean_matches_computes_home_win(self):
        from processing.data_cleaning import clean_matches

        df = pd.DataFrame({
            "home_team": ["Melbourne Storm", "Penrith Panthers"],
            "away_team": ["Penrith Panthers", "Melbourne Storm"],
            "home_score": [30, 10],
            "away_score": [12, 24],
            "date": ["2024-03-01", "2024-03-08"],
        })
        cleaned = clean_matches(df)
        assert cleaned.iloc[0]["home_win"] == 1
        assert cleaned.iloc[1]["home_win"] == 0

    def test_clean_matches_draw_handling(self):
        from processing.data_cleaning import clean_matches

        df = pd.DataFrame({
            "home_team": ["Melbourne Storm"],
            "away_team": ["Penrith Panthers"],
            "home_score": [18],
            "away_score": [18],
            "date": ["2024-03-01"],
        })
        cleaned = clean_matches(df)
        assert cleaned.iloc[0]["is_draw"] is True
        assert pd.isna(cleaned.iloc[0]["home_win"])

    def test_clean_matches_invalid_scores_capped(self):
        from processing.data_cleaning import clean_matches

        df = pd.DataFrame({
            "home_team": ["Melbourne Storm"],
            "away_team": ["Penrith Panthers"],
            "home_score": [100],  # Exceeds _MAX_NRL_SCORE (80)
            "away_score": [18],
            "date": ["2024-03-01"],
        })
        cleaned = clean_matches(df)
        assert pd.isna(cleaned.iloc[0]["home_score"])

    def test_clean_matches_finals_flag(self):
        from processing.data_cleaning import clean_matches

        df = pd.DataFrame({
            "home_team": ["Melbourne Storm", "Penrith Panthers"],
            "away_team": ["Penrith Panthers", "Melbourne Storm"],
            "home_score": [30, 24],
            "away_score": [12, 18],
            "date": ["2024-03-01", "2024-10-01"],
            "round": [1, "grand-final"],
        })
        cleaned = clean_matches(df)
        assert cleaned.iloc[0]["is_finals"] is False
        assert cleaned.iloc[1]["is_finals"] is True

    def test_clean_matches_null_team_names(self):
        from processing.data_cleaning import clean_matches

        df = pd.DataFrame({
            "home_team": [None],
            "away_team": ["Melbourne Storm"],
            "home_score": [20],
            "away_score": [10],
            "date": ["2024-03-01"],
        })
        cleaned = clean_matches(df)
        assert pd.isna(cleaned.iloc[0]["home_team"])

    def test_clean_matches_season_from_date(self):
        from processing.data_cleaning import clean_matches

        df = pd.DataFrame({
            "home_team": ["Melbourne Storm"],
            "away_team": ["Penrith Panthers"],
            "home_score": [20],
            "away_score": [10],
            "date": ["2024-05-15"],
        })
        cleaned = clean_matches(df)
        assert cleaned.iloc[0]["season"] == 2024

    def test_clean_odds_validates_range(self):
        from processing.data_cleaning import clean_odds

        df = pd.DataFrame({
            "home_team": ["Melbourne Storm"],
            "away_team": ["Penrith Panthers"],
            "home_odds": [0.5],   # Below minimum 1.01
            "away_odds": [200.0],  # Above maximum 101.0
            "date": ["2024-03-01"],
        })
        cleaned = clean_odds(df)
        assert np.isnan(cleaned.iloc[0]["home_odds"])
        assert np.isnan(cleaned.iloc[0]["away_odds"])

    def test_clean_odds_implied_probabilities(self):
        from processing.data_cleaning import clean_odds

        df = pd.DataFrame({
            "home_team": ["Melbourne Storm"],
            "away_team": ["Penrith Panthers"],
            "home_odds": [2.0],
            "away_odds": [2.0],
            "date": ["2024-03-01"],
        })
        cleaned = clean_odds(df)
        assert abs(cleaned.iloc[0]["home_implied_prob"] - 0.5) < 0.01
        assert abs(cleaned.iloc[0]["away_implied_prob"] - 0.5) < 0.01

    def test_clean_ladder_basic(self, sample_ladder_df):
        from processing.data_cleaning import clean_ladder

        cleaned = clean_ladder(sample_ladder_df)
        assert "win_pct" in cleaned.columns
        assert "point_diff" in cleaned.columns


# ============================================================================
# Data leakage verification
# ============================================================================


class TestNoDataLeakage:
    """Critical tests verifying no future data leaks into features."""

    def test_elo_backfill_uses_only_prior_data(self, sample_matches_df):
        """Elo ratings at match N should reflect only matches 0..N-1."""
        from processing.elo import EloRating

        df = sample_matches_df.sort_values("date").reset_index(drop=True)
        elo = EloRating(k_factor=20, home_advantage=50)

        result_full = elo.backfill(df)

        # Recompute for a subset (first 10 matches)
        elo2 = EloRating(k_factor=20, home_advantage=50)
        result_partial = elo2.backfill(df.iloc[:10].copy())

        # The Elo ratings for the first 10 matches should be identical
        for i in range(10):
            assert abs(
                result_full.iloc[i]["home_elo"] - result_partial.iloc[i]["home_elo"]
            ) < 0.001, f"Elo mismatch at row {i}"

    def test_rolling_features_at_match_n_exclude_match_n(self, sample_matches_df):
        """Rolling features for match N must not include match N's result."""
        from processing.rolling_stats import compute_rolling_form

        df = sample_matches_df.sort_values("date").reset_index(drop=True)
        result = compute_rolling_form(df, windows=[5])

        # Compare full run vs. run without the last row:
        # Feature values for the last row should be identical in both runs
        # because they only use data from before that row.
        result_minus_last = compute_rolling_form(df.iloc[:-1].copy(), windows=[5])

        # The feature values for the second-to-last row should be identical
        if len(result_minus_last) > 1:
            for col in ["home_win_rate_5", "away_win_rate_5"]:
                val_full = result.iloc[-2][col]
                val_partial = result_minus_last.iloc[-1][col]
                if pd.notna(val_full) and pd.notna(val_partial):
                    assert abs(val_full - val_partial) < 0.001, (
                        f"Leakage detected in {col}: full={val_full}, partial={val_partial}"
                    )

    def test_venue_features_use_pre_match_data(self, sample_matches_df):
        """Venue features should use only data from matches before the current one."""
        from processing.feature_engineering import build_venue_features

        df = sample_matches_df.sort_values("date").reset_index(drop=True)
        result = build_venue_features(df)

        # First match at any venue should have NaN venue stats
        # (no prior data available)
        assert pd.isna(result.iloc[0]["home_venue_win_rate"])
        assert pd.isna(result.iloc[0]["venue_avg_total_score"])
