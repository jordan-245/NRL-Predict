"""
Tests for backtesting framework: walk-forward splits, metrics, betting simulation.

Covers:
- WalkForwardBacktester: verify splits are temporal (no future data in train)
- Expanding window: train sets grow, test sets are single years
- Sliding window: fixed-width training window
- Metrics computation: accuracy, log_loss, brier_score, auc with known inputs
- BettingSimulator: flat stake and Kelly criterion calculations with known odds
- Backtest results have expected shape and columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Shared fixtures for backtest tests
# ============================================================================


@pytest.fixture()
def backtest_features_df() -> pd.DataFrame:
    """Feature DataFrame spanning years 2015-2022, suitable for backtesting."""
    np.random.seed(42)
    rows_per_year = 30
    years = list(range(2015, 2023))
    n = rows_per_year * len(years)

    year_col = []
    round_col = []
    for y in years:
        year_col.extend([y] * rows_per_year)
        round_col.extend(list(range(1, rows_per_year + 1)))

    return pd.DataFrame({
        "year": year_col,
        "round": round_col,
        "home_elo": np.random.normal(1500, 80, n),
        "away_elo": np.random.normal(1500, 80, n),
        "home_win_rate_5": np.random.uniform(0.2, 0.8, n),
        "away_win_rate_5": np.random.uniform(0.2, 0.8, n),
        "home_ladder_pos": np.random.randint(1, 17, n),
        "away_ladder_pos": np.random.randint(1, 17, n),
    })


@pytest.fixture()
def backtest_target(backtest_features_df) -> pd.Series:
    """Binary target aligned with backtest_features_df."""
    np.random.seed(42)
    return pd.Series(
        np.random.randint(0, 2, size=len(backtest_features_df)),
        name="home_win",
    )


class _SimpleModel:
    """Minimal model that always predicts home win with probability 0.6."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.ones(n, dtype=int)

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.column_stack([
            np.full(n, 0.4),
            np.full(n, 0.6),
        ])


# ============================================================================
# WalkForwardBacktester: split generation tests
# ============================================================================


class TestWalkForwardBacktesterSplits:
    """Test temporal split generation."""

    def test_expanding_window_splits_are_temporal(self, backtest_features_df):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2018, 2023),
            expanding=True,
            year_column="year",
        )

        for train_idx, test_idx, test_year in bt.get_splits(backtest_features_df):
            train_years = backtest_features_df.iloc[train_idx]["year"].values
            test_years = backtest_features_df.iloc[test_idx]["year"].values

            # All training years must be strictly before the test year
            assert (train_years < test_year).all(), (
                f"Training data contains year >= {test_year}"
            )
            # All test years must equal the test year
            assert (test_years == test_year).all()

    def test_expanding_window_train_grows(self, backtest_features_df):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2018, 2023),
            expanding=True,
            year_column="year",
        )

        train_sizes = []
        for train_idx, test_idx, test_year in bt.get_splits(backtest_features_df):
            train_sizes.append(len(train_idx))

        # With expanding window, train size should increase each fold
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], (
                f"Train size did not grow: {train_sizes[i-1]} -> {train_sizes[i]}"
            )

    def test_test_sets_are_single_years(self, backtest_features_df):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2018, 2023),
            expanding=True,
            year_column="year",
        )

        for train_idx, test_idx, test_year in bt.get_splits(backtest_features_df):
            test_years_actual = backtest_features_df.iloc[test_idx]["year"].unique()
            assert len(test_years_actual) == 1
            assert test_years_actual[0] == test_year

    def test_sliding_window_fixed_width(self, backtest_features_df):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2019, 2023),
            expanding=False,
            sliding_window_size=3,
            year_column="year",
        )

        for train_idx, test_idx, test_year in bt.get_splits(backtest_features_df):
            train_years = backtest_features_df.iloc[train_idx]["year"]
            min_year = train_years.min()
            max_year = train_years.max()

            # Sliding window: train should be at most `sliding_window_size` years wide
            # and end at test_year - 1
            assert max_year == test_year - 1
            # Min year should be max(start, test_year - window_size)
            expected_min = max(2015, test_year - 3)
            assert min_year == expected_min

    def test_no_future_data_in_training(self, backtest_features_df):
        """Critical: no data from the test year or later should appear in training."""
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2018, 2023),
            expanding=True,
            year_column="year",
        )

        for train_idx, test_idx, test_year in bt.get_splits(backtest_features_df):
            train_max_year = backtest_features_df.iloc[train_idx]["year"].max()
            assert train_max_year < test_year, (
                f"Training data contains year {train_max_year} "
                f"which is >= test year {test_year}"
            )

    def test_number_of_folds(self, backtest_features_df):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2018, 2023),
            expanding=True,
            year_column="year",
        )

        folds = list(bt.get_splits(backtest_features_df))
        assert len(folds) == 5  # 2018, 2019, 2020, 2021, 2022

    def test_missing_year_column_raises(self, backtest_features_df):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(year_column="nonexistent_column")
        with pytest.raises(ValueError, match="missing required column"):
            list(bt.get_splits(backtest_features_df))

    def test_empty_test_years_raises(self):
        from evaluation.backtesting import WalkForwardBacktester

        with pytest.raises(ValueError, match="at least one year"):
            WalkForwardBacktester(test_years=[])

    def test_train_and_test_indices_do_not_overlap(self, backtest_features_df):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2018, 2023),
            expanding=True,
            year_column="year",
        )

        for train_idx, test_idx, test_year in bt.get_splits(backtest_features_df):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Train and test overlap: {overlap}"


# ============================================================================
# WalkForwardBacktester: run tests
# ============================================================================


class TestWalkForwardBacktesterRun:
    """Tests for the full backtest run."""

    def test_run_returns_correct_types(
        self, backtest_features_df, backtest_target
    ):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2020, 2023),
            expanding=True,
            year_column="year",
        )

        results_df, predictions_df = bt.run(
            model_factory=_SimpleModel,
            features_df=backtest_features_df,
            target=backtest_target,
        )

        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(predictions_df, pd.DataFrame)

    def test_run_results_shape(self, backtest_features_df, backtest_target):
        from evaluation.backtesting import WalkForwardBacktester

        test_yrs = range(2020, 2023)
        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=test_yrs,
            expanding=True,
            year_column="year",
        )

        results_df, predictions_df = bt.run(
            model_factory=_SimpleModel,
            features_df=backtest_features_df,
            target=backtest_target,
        )

        # One row per test year
        assert len(results_df) == len(list(test_yrs))

    def test_run_results_columns(self, backtest_features_df, backtest_target):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2020, 2023),
            expanding=True,
            year_column="year",
        )

        results_df, predictions_df = bt.run(
            model_factory=_SimpleModel,
            features_df=backtest_features_df,
            target=backtest_target,
        )

        # Results should contain standard metrics
        expected_cols = {"accuracy", "log_loss", "brier_score", "n_train", "n_test"}
        assert expected_cols.issubset(set(results_df.columns))

    def test_run_predictions_columns(self, backtest_features_df, backtest_target):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2020, 2023),
            expanding=True,
            year_column="year",
        )

        results_df, predictions_df = bt.run(
            model_factory=_SimpleModel,
            features_df=backtest_features_df,
            target=backtest_target,
        )

        expected_cols = {"year", "y_true", "y_pred", "y_prob"}
        assert expected_cols.issubset(set(predictions_df.columns))

    def test_run_predictions_count(self, backtest_features_df, backtest_target):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2020, 2023),
            expanding=True,
            year_column="year",
        )

        results_df, predictions_df = bt.run(
            model_factory=_SimpleModel,
            features_df=backtest_features_df,
            target=backtest_target,
        )

        # Total predictions should equal the total test rows across all folds
        total_test = sum(
            (backtest_features_df["year"] == y).sum()
            for y in range(2020, 2023)
        )
        assert len(predictions_df) == total_test

    def test_run_no_retraining_model(self, backtest_features_df, backtest_target):
        from evaluation.backtesting import WalkForwardBacktester

        bt = WalkForwardBacktester(
            train_start_year=2015,
            test_years=range(2020, 2023),
            expanding=True,
            year_column="year",
        )

        results_df, predictions_df = bt.run(
            model_factory=_SimpleModel,
            features_df=backtest_features_df,
            target=backtest_target,
            needs_retraining=False,
        )

        assert len(results_df) > 0


# ============================================================================
# Metrics computation tests
# ============================================================================


class TestMetrics:
    """Tests for evaluation.metrics individual and aggregate metric functions."""

    def test_compute_accuracy_perfect(self):
        from evaluation.metrics import compute_accuracy

        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        assert compute_accuracy(y_true, y_pred) == 1.0

    def test_compute_accuracy_zero(self):
        from evaluation.metrics import compute_accuracy

        y_true = [1, 1, 1, 1]
        y_pred = [0, 0, 0, 0]
        assert compute_accuracy(y_true, y_pred) == 0.0

    def test_compute_accuracy_partial(self):
        from evaluation.metrics import compute_accuracy

        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]  # 2 correct out of 4
        assert abs(compute_accuracy(y_true, y_pred) - 0.5) < 0.01

    def test_compute_log_loss_perfect(self):
        from evaluation.metrics import compute_log_loss

        y_true = [1, 0, 1, 0]
        y_prob = [0.999, 0.001, 0.999, 0.001]
        ll = compute_log_loss(y_true, y_prob)
        assert ll < 0.01  # Near-perfect predictions have very low log-loss

    def test_compute_log_loss_worst(self):
        from evaluation.metrics import compute_log_loss

        y_true = [1, 0]
        y_prob = [0.01, 0.99]  # Very wrong
        ll = compute_log_loss(y_true, y_prob)
        assert ll > 1.0  # High log-loss for wrong predictions

    def test_compute_log_loss_clipping(self):
        """Ensure log-loss handles probabilities of 0 and 1 via clipping."""
        from evaluation.metrics import compute_log_loss

        y_true = [1, 0]
        y_prob = [1.0, 0.0]
        ll = compute_log_loss(y_true, y_prob)
        assert np.isfinite(ll)  # Should not be inf or nan

    def test_compute_brier_score_perfect(self):
        from evaluation.metrics import compute_brier_score

        y_true = [1, 0, 1, 0]
        y_prob = [1.0, 0.0, 1.0, 0.0]
        assert compute_brier_score(y_true, y_prob) == 0.0

    def test_compute_brier_score_worst(self):
        from evaluation.metrics import compute_brier_score

        y_true = [1, 0]
        y_prob = [0.0, 1.0]
        assert compute_brier_score(y_true, y_prob) == 1.0

    def test_compute_brier_score_range(self):
        from evaluation.metrics import compute_brier_score

        y_true = [1, 0, 1, 0, 1]
        y_prob = [0.8, 0.2, 0.7, 0.3, 0.6]
        brier = compute_brier_score(y_true, y_prob)
        assert 0.0 <= brier <= 1.0

    def test_compute_auc_roc_perfect(self):
        from evaluation.metrics import compute_auc_roc

        y_true = [1, 1, 0, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1]
        auc = compute_auc_roc(y_true, y_prob)
        assert auc == 1.0

    def test_compute_auc_roc_random(self):
        from evaluation.metrics import compute_auc_roc

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)
        auc = compute_auc_roc(y_true, y_prob)
        # Random predictions should give AUC around 0.5
        assert 0.3 <= auc <= 0.7

    def test_compute_auc_roc_single_class_returns_nan(self):
        from evaluation.metrics import compute_auc_roc

        y_true = [1, 1, 1, 1]
        y_prob = [0.9, 0.8, 0.7, 0.6]
        auc = compute_auc_roc(y_true, y_prob)
        assert np.isnan(auc)

    def test_compute_ece_basics(self):
        from evaluation.metrics import compute_ece

        y_true = [1, 0, 1, 0]
        y_prob = [0.9, 0.1, 0.8, 0.2]
        ece = compute_ece(y_true, y_prob, n_bins=5)
        assert 0.0 <= ece <= 1.0

    def test_compute_all_metrics_keys(self):
        from evaluation.metrics import compute_all_metrics

        y_true = [1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 1, 1, 0, 0]
        y_prob = [0.8, 0.2, 0.7, 0.6, 0.4, 0.3]
        result = compute_all_metrics(y_true, y_pred, y_prob)

        expected_keys = {"accuracy", "log_loss", "brier_score", "auc_roc", "ece"}
        assert expected_keys == set(result.keys())

    def test_compute_all_metrics_values_sensible(self):
        from evaluation.metrics import compute_all_metrics

        y_true = [1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
        result = compute_all_metrics(y_true, y_pred, y_prob)

        assert result["accuracy"] == 1.0
        assert result["log_loss"] < 0.2
        assert result["brier_score"] < 0.1
        assert result["auc_roc"] == 1.0

    def test_empty_arrays_raise(self):
        from evaluation.metrics import compute_accuracy

        with pytest.raises(ValueError, match="must not be empty"):
            compute_accuracy([], [])

    def test_mismatched_lengths_raise(self):
        from evaluation.metrics import compute_accuracy

        with pytest.raises(ValueError, match="length mismatch"):
            compute_accuracy([1, 0], [1])

    def test_compare_models(self):
        from evaluation.metrics import compare_models

        results = {
            "model_a": {"accuracy": 0.65, "log_loss": 0.68},
            "model_b": {"accuracy": 0.70, "log_loss": 0.60},
        }
        df = compare_models(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        # Should be sorted by accuracy descending
        assert df.index[0] == "model_b"


# ============================================================================
# Betting simulation tests
# ============================================================================


class TestBettingSimulator:
    """Tests for evaluation.betting_simulation.BettingSimulator."""

    def test_initial_bankroll(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0, unit_stake=10.0)
        assert sim.bankroll == 1000.0

    def test_invalid_bankroll_raises(self):
        from evaluation.betting_simulation import BettingSimulator

        with pytest.raises(ValueError, match="initial_bankroll must be positive"):
            BettingSimulator(initial_bankroll=0.0)

    def test_invalid_unit_stake_raises(self):
        from evaluation.betting_simulation import BettingSimulator

        with pytest.raises(ValueError, match="unit_stake must be positive"):
            BettingSimulator(initial_bankroll=1000.0, unit_stake=0.0)

    def test_flat_stake_value_bet_places_bets(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0, unit_stake=10.0)
        # Model thinks home wins at 0.65, bookmaker implies 0.50 -> edge = 0.15
        bets = sim.flat_stake_value_bet(
            y_prob_model=np.array([0.65]),
            odds_implied_prob=np.array([0.50]),
            odds_decimal=np.array([2.0]),
            y_true=np.array([1]),
            threshold=0.05,
        )
        assert len(bets) == 1
        assert bets[0].won is True

    def test_flat_stake_winning_bet_profit(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0, unit_stake=10.0)
        bets = sim.flat_stake_value_bet(
            y_prob_model=np.array([0.70]),
            odds_implied_prob=np.array([0.50]),
            odds_decimal=np.array([2.0]),
            y_true=np.array([1]),
            threshold=0.05,
        )
        bet = bets[0]
        # Stake = 10, odds = 2.0, won -> payout = 20, profit = 10
        assert bet.stake == 10.0
        assert bet.payout == 20.0
        assert bet.profit == 10.0
        assert sim.bankroll == 1010.0

    def test_flat_stake_losing_bet_loss(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0, unit_stake=10.0)
        bets = sim.flat_stake_value_bet(
            y_prob_model=np.array([0.70]),
            odds_implied_prob=np.array([0.50]),
            odds_decimal=np.array([2.0]),
            y_true=np.array([0]),  # Lost
            threshold=0.05,
        )
        bet = bets[0]
        assert bet.won is False
        assert bet.payout == 0.0
        assert bet.profit == -10.0
        assert sim.bankroll == 990.0

    def test_flat_stake_no_edge_no_bet(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0, unit_stake=10.0)
        bets = sim.flat_stake_value_bet(
            y_prob_model=np.array([0.50]),
            odds_implied_prob=np.array([0.50]),
            odds_decimal=np.array([2.0]),
            y_true=np.array([1]),
            threshold=0.05,
        )
        # Edge = 0.0, below threshold -> no bet
        assert len(bets) == 0
        assert sim.bankroll == 1000.0

    def test_kelly_criterion_positive_edge(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0)
        bets = sim.kelly_criterion_bet(
            y_prob_model=np.array([0.70]),
            odds_decimal=np.array([2.0]),
            y_true=np.array([1]),
            fraction=1.0,
        )
        # Kelly fraction: f* = (0.7 * 1.0 - 0.3) / 1.0 = 0.4
        # Full Kelly stake = 0.4 * 1000 = 400
        assert len(bets) == 1
        assert abs(bets[0].stake - 400.0) < 1.0

    def test_kelly_criterion_no_edge_no_bet(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0)
        bets = sim.kelly_criterion_bet(
            y_prob_model=np.array([0.40]),
            odds_decimal=np.array([2.0]),
            y_true=np.array([1]),
            fraction=0.25,
        )
        # Kelly: f* = (0.4 * 1.0 - 0.6) / 1.0 = -0.2 (negative -> no bet)
        assert len(bets) == 0

    def test_kelly_fractional_reduces_stake(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0)
        bets = sim.kelly_criterion_bet(
            y_prob_model=np.array([0.70]),
            odds_decimal=np.array([2.0]),
            y_true=np.array([1]),
            fraction=0.25,
        )
        # f* = 0.4, quarter Kelly = 0.25 * 0.4 * 1000 = 100
        assert len(bets) == 1
        assert abs(bets[0].stake - 100.0) < 1.0

    def test_reset_restores_initial_state(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0, unit_stake=10.0)
        sim.flat_stake_value_bet(
            y_prob_model=np.array([0.70]),
            odds_implied_prob=np.array([0.50]),
            odds_decimal=np.array([2.0]),
            y_true=np.array([0]),
            threshold=0.05,
        )
        assert sim.bankroll != 1000.0  # Changed after bet

        sim.reset()
        assert sim.bankroll == 1000.0
        assert len(sim.bets) == 0

    def test_get_summary_no_bets(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0)
        summary = sim.get_summary()
        assert summary["total_bets"] == 0
        assert summary["profit_loss"] == 0.0
        assert summary["roi"] == 0.0
        assert summary["final_bankroll"] == 1000.0

    def test_get_summary_with_bets(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0, unit_stake=10.0)
        sim.flat_stake_value_bet(
            y_prob_model=np.array([0.70, 0.65]),
            odds_implied_prob=np.array([0.50, 0.50]),
            odds_decimal=np.array([2.0, 2.0]),
            y_true=np.array([1, 0]),
            threshold=0.05,
        )
        summary = sim.get_summary()
        assert summary["total_bets"] == 2
        assert summary["wins"] == 1
        assert summary["losses"] == 1
        assert abs(summary["win_rate"] - 0.5) < 0.01
        expected_keys = {
            "total_bets", "wins", "losses", "win_rate",
            "total_staked", "total_payout", "profit_loss", "roi",
            "max_drawdown", "max_drawdown_pct", "sharpe_ratio", "final_bankroll",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_get_summary_roi_calculation(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0, unit_stake=100.0)
        # Place one winning bet: stake 100 at odds 2.0
        sim.flat_stake_value_bet(
            y_prob_model=np.array([0.70]),
            odds_implied_prob=np.array([0.50]),
            odds_decimal=np.array([2.0]),
            y_true=np.array([1]),
            threshold=0.05,
        )
        summary = sim.get_summary()
        # Profit = 100 (payout 200 - stake 100). ROI = 100/100 * 100 = 100%
        assert summary["roi"] == 100.0

    def test_bankroll_exhaustion_stops_betting(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=15.0, unit_stake=10.0)
        # First bet: lose 10, bankroll -> 5
        # Second bet: can't afford 10, should stop
        bets = sim.flat_stake_value_bet(
            y_prob_model=np.array([0.70, 0.70]),
            odds_implied_prob=np.array([0.50, 0.50]),
            odds_decimal=np.array([2.0, 2.0]),
            y_true=np.array([0, 1]),
            threshold=0.05,
        )
        assert len(bets) == 1  # Only first bet placed
        assert sim.bankroll == 5.0

    def test_simulation_input_length_mismatch_raises(self):
        from evaluation.betting_simulation import BettingSimulator

        sim = BettingSimulator(initial_bankroll=1000.0, unit_stake=10.0)
        with pytest.raises(ValueError, match="same length"):
            sim.flat_stake_value_bet(
                y_prob_model=np.array([0.70, 0.65]),
                odds_implied_prob=np.array([0.50]),  # Wrong length
                odds_decimal=np.array([2.0, 2.0]),
                y_true=np.array([1, 0]),
                threshold=0.05,
            )


# ============================================================================
# simulate_season convenience function tests
# ============================================================================


class TestSimulateSeason:
    """Tests for evaluation.betting_simulation.simulate_season."""

    def test_simulate_season_flat_value(self):
        from evaluation.betting_simulation import simulate_season

        df = pd.DataFrame({
            "y_true": [1, 0, 1, 0, 1],
            "y_prob": [0.70, 0.65, 0.60, 0.55, 0.75],
            "odds_decimal": [2.0, 2.0, 2.0, 2.0, 2.0],
        })
        result = simulate_season(df, strategy="flat_value", threshold=0.05)
        assert "total_bets" in result
        assert "profit_loss" in result
        assert "bets" in result

    def test_simulate_season_kelly(self):
        from evaluation.betting_simulation import simulate_season

        df = pd.DataFrame({
            "y_true": [1, 0, 1],
            "y_prob": [0.70, 0.65, 0.60],
            "odds_decimal": [2.0, 2.0, 2.0],
        })
        result = simulate_season(df, strategy="kelly", kelly_fraction=0.25)
        assert "total_bets" in result

    def test_simulate_season_missing_columns_raises(self):
        from evaluation.betting_simulation import simulate_season

        df = pd.DataFrame({"y_true": [1], "y_prob": [0.7]})  # Missing odds_decimal
        with pytest.raises(ValueError, match="missing required column"):
            simulate_season(df)

    def test_simulate_season_unknown_strategy_raises(self):
        from evaluation.betting_simulation import simulate_season

        df = pd.DataFrame({
            "y_true": [1],
            "y_prob": [0.7],
            "odds_decimal": [2.0],
        })
        with pytest.raises(ValueError, match="Unknown strategy"):
            simulate_season(df, strategy="invalid_strategy")

    def test_simulate_season_computes_implied_prob(self):
        from evaluation.betting_simulation import simulate_season

        df = pd.DataFrame({
            "y_true": [1],
            "y_prob": [0.70],
            "odds_decimal": [2.0],
        })
        # odds_implied_prob not provided -> should be computed as 1/odds
        result = simulate_season(df, strategy="flat_value", threshold=0.05)
        # Edge = 0.70 - 0.50 = 0.20 > 0.05, so bet should be placed
        assert result["total_bets"] == 1
