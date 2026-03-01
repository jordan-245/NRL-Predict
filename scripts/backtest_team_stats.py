#!/usr/bin/env python3
"""
Backtest: Team Season Stats Features
======================================
Walk-forward backtest comparing the baseline model (no team stats)
against models augmented with NRL.com team season stats (line breaks,
possession, tackle breaks, etc.).

For each game in year Y, the team's PRIOR season (Y-1) stats are used
as features — no look-ahead bias.

Tests:
  A. Baseline (current 197 features)
  B. Baseline + all team stats (~250 features)
  C. Baseline + top-N team stats (auto-selected by importance)
  D. Multiple blend weights with team stats

Also outputs CatBoost feature importance rankings.

Usage:
    python scripts/backtest_team_stats.py
    python scripts/backtest_team_stats.py --test-years 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from catboost import CatBoostClassifier

from config.settings import PROCESSED_DIR
from pipelines import v3, v4
from predict_round import (
    BEST_CAT_PARAMS, SAMPLE_WEIGHT_DECAY, FEATURE_COLS,
    load_historical_data, get_elo_params,
)

# ---------------------------------------------------------------------------
# Team stats loading and merging
# ---------------------------------------------------------------------------
TEAM_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "team_season_stats.parquet"

# Only use _average columns (per-game averages), not totals (which correlate
# with games played, not quality).
STATS_TO_USE = [
    "line_breaks_average",
    "tackle_breaks_average",
    "possession_pct_average",
    "set_completion_pct_average",
    "all_run_metres_average",
    "post_contact_metres_average",
    "offloads_average",
    "errors_average",
    "penalties_conceded_average",
    "missed_tackles_average",
    "ineffective_tackles_average",
    "handling_errors_average",
    "intercepts_average",
    "kick_return_metres_average",
    "points_average",
    "tries_average",
    "try_assists_average",
    "tackles_average",
    "conversion_pct_average",
    "line_engaged_average",
    "supports_average",
    "all_runs_average",
    "all_receipts_average",
    "goals_average",
    "decoy_runs_average",
    "dummy_half_runs_average",
]


def load_team_stats() -> pd.DataFrame:
    """Load team season stats (wide format: year, team, stat1, stat2, ...)."""
    if not TEAM_STATS_PATH.exists():
        print("  ⚠ team_season_stats.parquet not found. Run nrl_team_stats.py first.")
        return pd.DataFrame()
    df = pd.read_parquet(TEAM_STATS_PATH)
    return df


def merge_prior_season_stats(matches_df: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge prior-season team stats onto match rows.

    For a match in year Y, use each team's stats from year Y-1.
    Creates home_ts_* and away_ts_* columns, plus diff columns.
    """
    if team_stats.empty:
        return matches_df

    df = matches_df.copy()

    # Use only the average columns that exist
    available = [c for c in STATS_TO_USE if c in team_stats.columns]
    if not available:
        print("  ⚠ No team stats columns found!")
        return df

    # Create lookup: (year, team) → stats dict
    stats_cols = ["year", "team"] + available
    ts = team_stats[stats_cols].copy()

    # Shift year by +1 so we merge Y-1 stats onto year Y matches
    # For home team
    home_merge = ts[["year", "team"] + available].copy()
    home_merge["merge_year"] = home_merge["year"] + 1
    home_merge = home_merge.drop(columns=["year"])
    home_merge = home_merge.rename(columns={"team": "home_team", "merge_year": "year"})
    home_merge = home_merge.rename(columns={c: f"home_ts_{c}" for c in available})

    # For away team
    away_merge = ts[["year", "team"] + available].copy()
    away_merge["merge_year"] = away_merge["year"] + 1
    away_merge = away_merge.drop(columns=["year"])
    away_merge = away_merge.rename(columns={"team": "away_team", "merge_year": "year"})
    away_merge = away_merge.rename(columns={c: f"away_ts_{c}" for c in available})

    # Merge onto matches
    pre_len = len(df)
    df = df.merge(home_merge, on=["year", "home_team"], how="left")
    df = df.merge(away_merge, on=["year", "away_team"], how="left")
    assert len(df) == pre_len, f"Merge changed row count: {pre_len} → {len(df)}"

    # Compute diff features (home - away) for each stat
    diff_cols = []
    for stat in available:
        h_col = f"home_ts_{stat}"
        a_col = f"away_ts_{stat}"
        d_col = f"ts_diff_{stat}"
        if h_col in df.columns and a_col in df.columns:
            df[d_col] = df[h_col] - df[a_col]
            diff_cols.append(d_col)

    n_home = sum(1 for c in df.columns if c.startswith("home_ts_"))
    n_away = sum(1 for c in df.columns if c.startswith("away_ts_"))
    n_diff = len(diff_cols)
    coverage = df[[f"home_ts_{available[0]}"]].notna().mean().values[0] * 100

    print(f"  Added {n_home + n_away + n_diff} team stats features "
          f"({n_home} home + {n_away} away + {n_diff} diff), "
          f"{coverage:.0f}% coverage")

    return df


# ---------------------------------------------------------------------------
# Feature building (reuse from backtest_strategy.py)
# ---------------------------------------------------------------------------

def build_all_features(matches, ladders, odds, elo_params):
    """Build V4 features for ALL matches (no upcoming split)."""
    linked = v3.link_odds(matches, odds)
    linked = linked.dropna(subset=["home_score"]).reset_index(drop=True)
    linked = linked[linked["home_score"] != linked["away_score"]].reset_index(drop=True)

    all_m = linked.sort_values("date").reset_index(drop=True)

    all_m = v3.backfill_elo(all_m, elo_params)
    all_m = v3.compute_rolling_form_features(all_m)
    all_m = v3.compute_h2h_features(all_m)
    all_m = v3.compute_ladder_features(all_m, ladders)
    all_m = v3.compute_venue_features(all_m)
    all_m = v3.compute_odds_features(all_m)
    all_m = v3.compute_schedule_features(all_m)
    all_m = v3.compute_contextual_features(all_m)
    all_m = v3.compute_engineered_features(all_m)

    all_m = v4.compute_v4_odds_features(all_m)
    all_m = v4.compute_scoring_consistency_features(all_m)
    all_m = v4.compute_attendance_features(all_m)
    all_m = v4.compute_kickoff_features(all_m)
    all_m = v4.compute_lineup_stability_features(all_m)
    all_m = v4.compute_player_impact_features(all_m)
    all_m = v4.compute_v4_engineered_features(all_m)

    all_m["home_win"] = np.where(
        all_m["home_score"] > all_m["away_score"], 1.0,
        np.where(all_m["home_score"] < all_m["away_score"], 0.0, np.nan)
    )

    return all_m


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def select_top_features(X_train, y_train, sample_weights, all_feature_cols, top_n=50):
    """Train a quick CatBoost and return top_n features by importance."""
    quick_params = dict(BEST_CAT_PARAMS)
    quick_params["iterations"] = 300
    quick_params["verbose"] = 0

    model = CatBoostClassifier(**quick_params)
    model.fit(X_train[all_feature_cols], y_train, sample_weight=sample_weights)
    imp = model.get_feature_importance()
    ranked = sorted(zip(all_feature_cols, imp), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_n]], ranked


def train_catboost(X_train, y_train, sample_weights, feature_cols):
    """Train CatBoost classifier and return model + probabilities."""
    params = dict(BEST_CAT_PARAMS)
    params["verbose"] = 0

    model = CatBoostClassifier(**params)
    model.fit(X_train[feature_cols], y_train, sample_weight=sample_weights)
    return model


def get_blend_accuracy(model_probs, odds_probs, y_true, model_weight):
    """Compute accuracy for a given model/odds blend weight."""
    blend = model_weight * model_probs + (1 - model_weight) * odds_probs
    preds = (blend > 0.5).astype(float)
    return (preds == y_true).mean()


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def run_backtest(test_years, min_train_year=2014):
    """Run the full walk-forward backtest comparing baseline vs team-stats models."""

    print("\n" + "=" * 70)
    print("  LOADING DATA")
    print("=" * 70)

    matches, ladders, odds = load_historical_data()
    elo_params = get_elo_params(matches, retune=False)

    print("\n  Building V4 features...")
    all_data = build_all_features(matches, ladders, odds, elo_params)

    # Merge team stats
    print("\n  Loading team season stats...")
    team_stats = load_team_stats()
    all_data = merge_prior_season_stats(all_data, team_stats)

    # Define feature sets
    baseline_cols = [c for c in FEATURE_COLS if c in all_data.columns]

    # Team stats columns (newly added)
    ts_cols = [c for c in all_data.columns
               if c.startswith("home_ts_") or c.startswith("away_ts_") or c.startswith("ts_diff_")]
    all_cols = baseline_cols + ts_cols

    # Ensure numeric
    for col in all_cols:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")

    print(f"\n  Baseline features: {len(baseline_cols)}")
    print(f"  Team stats features: {len(ts_cols)}")
    print(f"  Combined features: {len(all_cols)}")

    # ---------------------------------------------------------------------------
    # Walk-forward loop
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  WALK-FORWARD BACKTEST")
    print("=" * 70)

    results = []
    all_importance = []
    blend_weights = [0.0, 0.10, 0.20, 0.28, 0.35, 0.45, 0.55, 0.65]

    for test_year in test_years:
        t0 = time.time()

        # Train/test split
        train_mask = (all_data["year"] >= min_train_year) & (all_data["year"] < test_year)
        test_mask = all_data["year"] == test_year

        train_df = all_data[train_mask].copy()
        test_df = all_data[test_mask].copy()

        if len(train_df) < 100 or len(test_df) < 10:
            print(f"\n  {test_year}: skipped (train={len(train_df)}, test={len(test_df)})")
            continue

        y_train = train_df["home_win"].values
        y_test = test_df["home_win"].values

        # Sample weights
        train_years = train_df["year"].values
        max_yr = train_years.max()
        sample_weights = SAMPLE_WEIGHT_DECAY ** (max_yr - train_years)

        # Fill missing
        bool_cols_set = {"home_is_back_to_back", "away_is_back_to_back",
                         "home_bye_last_round", "away_bye_last_round",
                         "is_finals", "odds_home_favourite", "is_home", "is_neutral_venue"}

        for col_set in [baseline_cols, all_cols]:
            for c in col_set:
                if c in bool_cols_set:
                    train_df[c] = train_df[c].fillna(0)
                    test_df[c] = test_df[c].fillna(0)

        # Median fill for train, then apply same medians to test
        train_medians = train_df[all_cols].median()
        train_df[all_cols] = train_df[all_cols].fillna(train_medians)
        test_df[all_cols] = test_df[all_cols].fillna(train_medians)

        # Odds probs
        odds_probs_test = test_df["odds_home_prob"].values

        # ===================================================================
        # A. BASELINE (top-50 from baseline features only)
        # ===================================================================
        _, base_ranked = select_top_features(
            train_df, y_train, sample_weights, baseline_cols, top_n=50
        )
        base_top50 = [c for c, _ in base_ranked[:50]]
        model_a = train_catboost(train_df, y_train, sample_weights, base_top50)
        probs_a = model_a.predict_proba(test_df[base_top50])[:, 1]

        # ===================================================================
        # B. ALL features (top-50 from baseline + team stats)
        # ===================================================================
        _, all_ranked = select_top_features(
            train_df, y_train, sample_weights, all_cols, top_n=50
        )
        all_top50 = [c for c, _ in all_ranked[:50]]
        model_b = train_catboost(train_df, y_train, sample_weights, all_top50)
        probs_b = model_b.predict_proba(test_df[all_top50])[:, 1]

        # Save importance from the combined model
        for rank, (feat, imp) in enumerate(all_ranked):
            all_importance.append({
                "year": test_year,
                "feature": feat,
                "importance": imp,
                "rank": rank + 1,
                "is_team_stat": feat in ts_cols,
            })

        # ===================================================================
        # C. TOP-50 with forced team stats (top-35 baseline + top-15 team stats)
        # ===================================================================
        base_only_ranked = [(c, i) for c, i in all_ranked if c not in ts_cols][:35]
        ts_only_ranked = [(c, i) for c, i in all_ranked if c in ts_cols][:15]
        hybrid_features = [c for c, _ in base_only_ranked + ts_only_ranked]
        model_c = train_catboost(train_df, y_train, sample_weights, hybrid_features)
        probs_c = model_c.predict_proba(test_df[hybrid_features])[:, 1]

        # ===================================================================
        # D. TOP-60 features (larger feature set)
        # ===================================================================
        all_top60 = [c for c, _ in all_ranked[:60]]
        model_d = train_catboost(train_df, y_train, sample_weights, all_top60)
        probs_d = model_d.predict_proba(test_df[all_top60])[:, 1]

        # ===================================================================
        # E. TEAM STATS ONLY (just the ts_ columns, top-25)
        # ===================================================================
        ts_available = [c for c in ts_cols if c in train_df.columns]
        if len(ts_available) >= 10:
            _, ts_ranked = select_top_features(
                train_df, y_train, sample_weights, ts_available, top_n=25
            )
            ts_top25 = [c for c, _ in ts_ranked[:25]]
            model_e = train_catboost(train_df, y_train, sample_weights, ts_top25)
            probs_e = model_e.predict_proba(test_df[ts_top25])[:, 1]
        else:
            probs_e = np.full(len(test_df), 0.5)

        # ===================================================================
        # Evaluate all strategies across blend weights
        # ===================================================================
        n_test = len(test_df)
        elapsed = time.time() - t0

        # Count team stats in top-50
        ts_in_top50 = sum(1 for c, _ in all_ranked[:50] if c in ts_cols)
        ts_top5 = [(c, round(i, 1)) for c, i in all_ranked[:5] if c in ts_cols]

        print(f"\n  {test_year} ({n_test} games, {elapsed:.0f}s)")
        print(f"    Team stats in top-50: {ts_in_top50}/50")
        if ts_top5:
            print(f"    Team stats in top-5: {ts_top5}")

        # Header
        print(f"\n    {'Strategy':<25s}", end="")
        for bw in blend_weights:
            label = f"M{int(bw*100)}/O{int((1-bw)*100)}"
            print(f" {label:>9s}", end="")
        print(f" {'Model%':>7s}")

        # Compute results for each strategy and blend weight
        strategies = {
            "A: Baseline top50": probs_a,
            "B: +TeamStats top50": probs_b,
            "C: Hybrid 35+15": probs_c,
            "D: +TeamStats top60": probs_d,
            "E: TeamStats only": probs_e,
        }

        for strat_name, model_probs in strategies.items():
            print(f"    {strat_name:<25s}", end="")
            model_only_acc = ((model_probs > 0.5).astype(float) == y_test).mean()

            for bw in blend_weights:
                if bw == 0.0:
                    acc = (odds_probs_test > 0.5).astype(float)
                    acc = (acc == y_test).mean()
                else:
                    acc = get_blend_accuracy(model_probs, odds_probs_test, y_test, bw)
                print(f" {acc:>8.1%}", end="")

                results.append({
                    "year": test_year,
                    "strategy": strat_name,
                    "blend_weight": bw,
                    "accuracy": acc,
                    "n_games": n_test,
                    "correct": int(acc * n_test),
                    "model_only_acc": model_only_acc,
                })

            print(f" {model_only_acc:>6.1%}")

    # ---------------------------------------------------------------------------
    # Summary across all years
    # ---------------------------------------------------------------------------
    df_res = pd.DataFrame(results)
    df_imp = pd.DataFrame(all_importance)

    print("\n" + "=" * 70)
    print("  OVERALL RESULTS (all test years combined)")
    print("=" * 70)

    # Overall accuracy per strategy x blend
    summary = df_res.groupby(["strategy", "blend_weight"]).agg(
        total_games=("n_games", "sum"),
        total_correct=("correct", "sum"),
    ).reset_index()
    summary["accuracy"] = summary["total_correct"] / summary["total_games"]

    # Find best blend per strategy
    best_per_strat = summary.loc[summary.groupby("strategy")["accuracy"].idxmax()]

    print(f"\n  {'Strategy':<25s} {'Best Blend':>12s} {'Accuracy':>10s} {'Correct':>9s} {'Total':>7s} {'vs Odds':>9s}")
    print(f"  {'-'*75}")

    odds_baseline = summary[(summary["strategy"] == "A: Baseline top50") &
                            (summary["blend_weight"] == 0.0)]
    odds_correct = odds_baseline["total_correct"].values[0] if len(odds_baseline) else 0
    odds_total = odds_baseline["total_games"].values[0] if len(odds_baseline) else 1

    for _, row in best_per_strat.sort_values("accuracy", ascending=False).iterrows():
        bw = row["blend_weight"]
        label = f"M{int(bw*100)}/O{int((1-bw)*100)}" if bw > 0 else "Pure odds"
        delta = row["total_correct"] - odds_correct
        sign = "+" if delta >= 0 else ""
        print(f"  {row['strategy']:<25s} {label:>12s} {row['accuracy']:>9.1%} "
              f"{int(row['total_correct']):>8d} {int(row['total_games']):>7d} {sign}{delta:>8.0f}")

    # ---------------------------------------------------------------------------
    # Feature importance analysis
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FEATURE IMPORTANCE (averaged across all test years)")
    print("=" * 70)

    avg_imp = df_imp.groupby(["feature", "is_team_stat"]).agg(
        mean_importance=("importance", "mean"),
        mean_rank=("rank", "mean"),
        min_rank=("rank", "min"),
        max_rank=("rank", "max"),
    ).reset_index().sort_values("mean_importance", ascending=False)

    print(f"\n  Top 30 features overall:")
    print(f"  {'Rank':>4s} {'Feature':<45s} {'Importance':>10s} {'Avg Rank':>9s} {'TS?':>4s}")
    print(f"  {'-'*75}")
    for i, (_, row) in enumerate(avg_imp.head(30).iterrows(), 1):
        ts_flag = "★" if row["is_team_stat"] else ""
        print(f"  {i:>4d} {row['feature']:<45s} {row['mean_importance']:>10.1f} "
              f"{row['mean_rank']:>8.1f} {ts_flag:>4s}")

    # Team stats only
    ts_imp = avg_imp[avg_imp["is_team_stat"]]
    if len(ts_imp):
        print(f"\n  Top 15 TEAM STATS features:")
        print(f"  {'Rank':>4s} {'Feature':<45s} {'Importance':>10s} {'Avg Rank':>9s}")
        print(f"  {'-'*70}")
        for i, (_, row) in enumerate(ts_imp.head(15).iterrows(), 1):
            print(f"  {i:>4d} {row['feature']:<45s} {row['mean_importance']:>10.1f} "
                  f"{row['mean_rank']:>8.1f}")

    # ---------------------------------------------------------------------------
    # Per-year best strategy
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  BEST STRATEGY PER YEAR")
    print("=" * 70)

    for yr in sorted(df_res["year"].unique()):
        yr_data = df_res[df_res["year"] == yr]
        best = yr_data.loc[yr_data["accuracy"].idxmax()]
        bw = best["blend_weight"]
        label = f"M{int(bw*100)}/O{int((1-bw)*100)}" if bw > 0 else "Pure odds"
        print(f"  {yr}: {best['strategy']:<25s} {label:<12s} "
              f"{best['accuracy']:.1%} ({int(best['correct'])}/{int(best['n_games'])})")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    out_dir = PROJECT_ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_res.to_csv(out_dir / "team_stats_backtest_results.csv", index=False)
    avg_imp.to_csv(out_dir / "team_stats_feature_importance.csv", index=False)
    print(f"\n  Saved results to {out_dir}/team_stats_backtest_results.csv")
    print(f"  Saved importance to {out_dir}/team_stats_feature_importance.csv")

    return df_res, df_imp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest team season stats features")
    parser.add_argument("--test-years", nargs="+", type=int,
                        default=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--min-train-year", type=int, default=2014,
                        help="Earliest training year (default: 2014, since stats start 2013)")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  NRL Team Stats Feature Backtest")
    print(f"  Test years: {args.test_years}")
    print(f"  Training from: {args.min_train_year}")
    print("=" * 70)

    run_backtest(args.test_years, args.min_train_year)


if __name__ == "__main__":
    main()
