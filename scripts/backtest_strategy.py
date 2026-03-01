#!/usr/bin/env python3
"""
Comprehensive Strategy Backtest
================================
Walk-forward backtest: for each season Y (2018-2025), train on all prior
data and predict season Y.  Tests multiple blend strategies, late-odds
refresh, and confidence tiers.

Strategies tested:
  1. Pure odds favourite (baseline)
  2. CatBoost model only (no odds)
  3. Model+Odds blend at various weights (10/90 to 60/40)
  4. Tiered: odds for LOCKs, model-heavy for close games
  5. Late odds refresh (open→close) for close games
  6. Adaptive: model weight increases for close games

Each strategy evaluated per-season and overall.

Usage:
    python scripts/backtest_strategy.py
    python scripts/backtest_strategy.py --test-years 2023 2024 2025
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
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config.settings import PROCESSED_DIR
from pipelines import v3, v4
from predict_round import (
    BEST_CAT_PARAMS, SAMPLE_WEIGHT_DECAY, FEATURE_COLS,
    load_historical_data, _sanitize_round1_features, _fill_odds_coherent,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-years", nargs="+", type=int,
                   default=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    p.add_argument("--min-train-year", type=int, default=2013)
    return p.parse_args()


def build_all_features(matches, ladders, odds, elo_params):
    """Build V4 features for ALL matches (no train/test split)."""
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
    all_m = v4.compute_team_stats_features(all_m)
    all_m = v4.compute_referee_features(all_m)
    all_m = v4.compute_v4_engineered_features(all_m)

    all_m["home_win"] = np.where(
        all_m["home_score"] > all_m["away_score"], 1.0,
        np.where(all_m["home_score"] < all_m["away_score"], 0.0, np.nan)
    )

    feature_cols = [c for c in FEATURE_COLS if c in all_m.columns]
    for col in feature_cols:
        all_m[col] = pd.to_numeric(all_m[col], errors="coerce")

    return all_m, feature_cols


def walk_forward_predictions(all_data, feature_cols, test_years, min_train_year):
    """Walk-forward: for each test year, train on prior years, predict test year."""
    results = []

    for test_year in test_years:
        t0 = time.time()

        train_mask = (all_data["year"] >= min_train_year) & (all_data["year"] < test_year)
        test_mask = all_data["year"] == test_year

        train_df = all_data[train_mask].copy()
        test_df = all_data[test_mask].copy()

        if len(train_df) < 100 or len(test_df) < 10:
            print(f"  {test_year}: skipped (train={len(train_df)}, test={len(test_df)})")
            continue

        X_train_raw = train_df[feature_cols].copy()
        y_train = train_df["home_win"].values
        X_test_raw = test_df[feature_cols].copy()
        y_test = test_df["home_win"].values

        # Sample weights
        train_years = train_df["year"].values
        max_yr = train_years.max()
        sample_weights = SAMPLE_WEIGHT_DECAY ** (max_yr - train_years)

        # Fill missing
        bool_cols = {"home_is_back_to_back", "away_is_back_to_back",
                     "home_bye_last_round", "away_bye_last_round",
                     "is_finals", "odds_home_favourite", "is_home", "is_neutral_venue"}
        medians = X_train_raw.median()

        X_train = _sanitize_round1_features(X_train_raw.copy())
        X_train = _fill_odds_coherent(X_train)
        X_test = _sanitize_round1_features(X_test_raw.copy())
        X_test = _fill_odds_coherent(X_test)

        for col in feature_cols:
            med = medians.get(col, 0)
            if pd.isna(med):
                med = 0
            fill_val = 0 if col in bool_cols else med
            X_train[col] = X_train[col].fillna(fill_val)
            X_test[col] = X_test[col].fillna(fill_val)

        # Feature selection — top 50
        selector = xgb.XGBClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.02, verbosity=0,
                                      random_state=42)
        selector.fit(X_train, y_train, sample_weight=sample_weights)
        imp = pd.Series(selector.feature_importances_, index=feature_cols)
        top50 = list(imp.sort_values(ascending=False).head(50).index)

        # Train CatBoost on top-50
        model = CatBoostClassifier(**BEST_CAT_PARAMS)
        model.fit(X_train[top50], y_train, sample_weight=sample_weights)
        model_probs = np.clip(model.predict_proba(X_test[top50])[:, 1], 1e-7, 1-1e-7)

        # Collect per-game results
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            odds_prob = row.get("odds_home_prob", np.nan)
            if pd.isna(odds_prob) and "h2h_home" in row and "h2h_away" in row:
                h, a = row["h2h_home"], row["h2h_away"]
                if pd.notna(h) and pd.notna(a) and h > 0 and a > 0:
                    odds_prob = (1/h) / (1/h + 1/a)

            # Open/close odds for late-odds test
            open_prob = np.nan
            close_prob = np.nan
            if "h2h_home_open" in row and pd.notna(row.get("h2h_home_open")):
                ho, ao = row["h2h_home_open"], row.get("h2h_away_open", np.nan)
                if pd.notna(ho) and pd.notna(ao) and ho > 0 and ao > 0:
                    open_prob = (1/ho) / (1/ho + 1/ao)
            if "h2h_home_close" in row and pd.notna(row.get("h2h_home_close")):
                hc, ac = row["h2h_home_close"], row.get("h2h_away_close", np.nan)
                if pd.notna(hc) and pd.notna(ac) and hc > 0 and ac > 0:
                    close_prob = (1/hc) / (1/hc + 1/ac)

            results.append({
                "year": int(row["year"]),
                "round": row["round"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_win": int(row["home_win"]),
                "model_prob": float(model_probs[i]),
                "odds_prob": float(odds_prob) if pd.notna(odds_prob) else np.nan,
                "open_prob": float(open_prob) if pd.notna(open_prob) else np.nan,
                "close_prob": float(close_prob) if pd.notna(close_prob) else np.nan,
            })

        elapsed = time.time() - t0
        n_test = len(test_df)
        model_acc = (np.round(model_probs) == y_test).mean()
        print(f"  {test_year}: {n_test} games, model acc {model_acc:.1%} ({elapsed:.1f}s)")

    return pd.DataFrame(results)


def evaluate_strategies(df):
    """Evaluate multiple tipping strategies on walk-forward predictions."""
    # Filter to games with odds
    has_odds = df[df["odds_prob"].notna()].copy()
    n = len(has_odds)
    hw = has_odds["home_win"].values

    print(f"\n{'='*75}")
    print(f"  STRATEGY COMPARISON — {n} games with odds")
    print(f"{'='*75}")

    strategies = {}

    # ── S1: Pure odds favourite ──
    tips = (has_odds["odds_prob"].values >= 0.5).astype(int)
    correct = (tips == hw).sum()
    strategies["S1: Pure odds favourite"] = correct
    baseline = correct

    # ── S2: Model only ──
    tips = (has_odds["model_prob"].values >= 0.5).astype(int)
    correct = (tips == hw).sum()
    strategies["S2: Model only (CatBoost)"] = correct

    # ── S3: Different blend weights ──
    for model_w in [0.10, 0.20, 0.30, 0.40, 0.495, 0.60, 0.70]:
        odds_w = 1.0 - model_w
        blend = model_w * has_odds["model_prob"].values + odds_w * has_odds["odds_prob"].values
        tips = (blend >= 0.5).astype(int)
        correct = (tips == hw).sum()
        strategies[f"S3: Blend {model_w:.0%}m/{odds_w:.0%}o"] = correct

    # ── S4: Tiered — odds for LOCKs, heavy model for close ──
    for lock_thresh in [0.60, 0.65, 0.70]:
        odds_p = has_odds["odds_prob"].values
        model_p = has_odds["model_prob"].values
        confidence = np.abs(odds_p - 0.5) * 2

        # LOCKs: pure odds. Close games: 70% model / 30% odds
        is_lock = confidence >= (lock_thresh - 0.5) * 2
        blend = np.where(
            is_lock,
            odds_p,
            0.70 * model_p + 0.30 * odds_p
        )
        tips = (blend >= 0.5).astype(int)
        correct = (tips == hw).sum()
        strategies[f"S4: Tiered (lock>{lock_thresh:.0%}, close=70m/30o)"] = correct

    # ── S5: Late odds refresh for close games ──
    has_both = has_odds[has_odds["open_prob"].notna() & has_odds["close_prob"].notna()].copy()
    if len(has_both) > 50:
        n5 = len(has_both)
        hw5 = has_both["home_win"].values
        open_p = has_both["open_prob"].values
        close_p = has_both["close_prob"].values
        model_p5 = has_both["model_prob"].values

        # S5a: Model blend with opening odds (Tuesday)
        blend_open = 0.495 * model_p5 + 0.505 * open_p
        tips_open = (blend_open >= 0.5).astype(int)
        correct_open = (tips_open == hw5).sum()

        # S5b: Selective late refresh — re-blend close games with closing odds
        for thresh in [0.55, 0.60, 0.65]:
            confidence = np.abs(blend_open - 0.5) * 2
            is_close = confidence < (thresh - 0.5) * 2
            blend_refresh = np.where(
                is_close,
                0.495 * model_p5 + 0.505 * close_p,
                blend_open
            )
            tips_refresh = (blend_refresh >= 0.5).astype(int)
            correct_refresh = (tips_refresh == hw5).sum()
            flips = (tips_refresh != tips_open).sum()
            strategies[f"S5: Late refresh <{thresh:.0%} ({n5}g)"] = \
                f"{correct_refresh}/{n5} ({correct_refresh/n5:.1%}) Δ{correct_refresh-correct_open:+d}, {flips} flips"

        strategies[f"S5: Open blend baseline ({n5}g)"] = \
            f"{correct_open}/{n5} ({correct_open/n5:.1%})"

    # ── S6: Adaptive model weight — heavier for close games ──
    odds_p = has_odds["odds_prob"].values
    model_p = has_odds["model_prob"].values
    # Model weight: 0.3 for LOCKs, scales up to 0.7 for coin flips
    confidence = np.abs(odds_p - 0.5) * 2  # 0=tossup, 1=certain
    adaptive_model_w = 0.7 - 0.4 * confidence  # 0.7 at tossup, 0.3 at certain
    adaptive_odds_w = 1.0 - adaptive_model_w
    blend = adaptive_model_w * model_p + adaptive_odds_w * odds_p
    tips = (blend >= 0.5).astype(int)
    correct = (tips == hw).sum()
    strategies["S6: Adaptive (0.7m@tossup, 0.3m@lock)"] = correct

    # More adaptive variants
    for base_w, close_w in [(0.20, 0.60), (0.30, 0.70), (0.10, 0.80)]:
        confidence = np.abs(odds_p - 0.5) * 2
        model_w = close_w - (close_w - base_w) * confidence
        blend = model_w * model_p + (1.0 - model_w) * odds_p
        tips = (blend >= 0.5).astype(int)
        correct = (tips == hw).sum()
        strategies[f"S6: Adaptive ({close_w:.0%}m@tossup, {base_w:.0%}m@lock)"] = correct

    # ── S7: Optimised fixed blend (find best weight) ──
    best_w = 0
    best_correct = 0
    for w in np.arange(0.0, 1.01, 0.01):
        blend = w * model_p + (1.0 - w) * odds_p
        tips = (blend >= 0.5).astype(int)
        c = (tips == hw).sum()
        if c > best_correct:
            best_correct = c
            best_w = w
    strategies[f"S7: Optimal fixed blend ({best_w:.0%}m/{1-best_w:.0%}o)"] = best_correct

    # Print results
    print(f"\n  {'Strategy':<52s} {'Correct':>8s} {'Acc':>7s} {'Δ':>5s}")
    print(f"  {'─'*75}")
    for name, val in strategies.items():
        if isinstance(val, str):
            print(f"  {name:<52s} {val}")
        else:
            acc = val / n
            delta = val - baseline
            numeric_vals = [v for v in strategies.values() if isinstance(v, (int, float, np.integer))]
            marker = " ◄◄" if numeric_vals and val == max(numeric_vals) else ""
            print(f"  {name:<52s} {val:>8d} {acc:>7.1%} {delta:>+5d}{marker}")

    return strategies, baseline, has_odds


def per_year_breakdown(df, best_strategy_name="S3: Blend 50%m/50%o"):
    """Show per-year accuracy for key strategies."""
    has_odds = df[df["odds_prob"].notna()].copy()

    print(f"\n{'='*75}")
    print(f"  PER-YEAR BREAKDOWN")
    print(f"{'='*75}")
    print(f"\n  {'Year':>6s} {'N':>5s} {'Odds':>6s} {'Model':>7s} "
          f"{'50/50':>6s} {'Adapt':>6s} {'Best':>6s}")
    print(f"  {'─'*55}")

    for yr in sorted(has_odds["year"].unique()):
        ym = has_odds[has_odds["year"] == yr]
        n = len(ym)
        hw = ym["home_win"].values
        odds_p = ym["odds_prob"].values
        model_p = ym["model_prob"].values

        odds_ok = ((odds_p >= 0.5).astype(int) == hw).sum()
        model_ok = ((model_p >= 0.5).astype(int) == hw).sum()

        # 50/50 blend
        b50 = 0.495 * model_p + 0.505 * odds_p
        b50_ok = ((b50 >= 0.5).astype(int) == hw).sum()

        # Adaptive
        conf = np.abs(odds_p - 0.5) * 2
        adapt_w = 0.70 - 0.40 * conf
        ba = adapt_w * model_p + (1.0 - adapt_w) * odds_p
        adapt_ok = ((ba >= 0.5).astype(int) == hw).sum()

        best = max(odds_ok, model_ok, b50_ok, adapt_ok)

        print(f"  {yr:>6d} {n:>5d} {odds_ok:>6d} {model_ok:>7d} "
              f"{b50_ok:>6d} {adapt_ok:>6d} {best:>6d}")

    # Also show confidence-tier breakdown
    print(f"\n{'='*75}")
    print(f"  ACCURACY BY CONFIDENCE TIER")
    print(f"{'='*75}")
    odds_p = has_odds["odds_prob"].values
    model_p = has_odds["model_prob"].values
    hw = has_odds["home_win"].values

    for label, lo, hi in [("LOCK (>65%)", 0.65, 1.0),
                           ("LEAN (55-65%)", 0.55, 0.65),
                           ("TOSS-UP (<55%)", 0.0, 0.55)]:
        odds_conf = np.maximum(odds_p, 1 - odds_p)
        mask = (odds_conf >= lo) & (odds_conf < hi)
        n_tier = mask.sum()
        if n_tier == 0:
            continue

        tier_hw = hw[mask]
        # Odds
        odds_ok = ((odds_p[mask] >= 0.5).astype(int) == tier_hw).sum()
        # Model
        model_ok = ((model_p[mask] >= 0.5).astype(int) == tier_hw).sum()
        # 50/50
        b = 0.495 * model_p[mask] + 0.505 * odds_p[mask]
        blend_ok = ((b >= 0.5).astype(int) == tier_hw).sum()
        # Adaptive
        c = np.abs(odds_p[mask] - 0.5) * 2
        aw = 0.70 - 0.40 * c
        ba = aw * model_p[mask] + (1.0 - aw) * odds_p[mask]
        adapt_ok = ((ba >= 0.5).astype(int) == tier_hw).sum()

        print(f"\n  {label} ({n_tier} games):")
        print(f"    Odds:     {odds_ok:>4d}/{n_tier} ({odds_ok/n_tier:.1%})")
        print(f"    Model:    {model_ok:>4d}/{n_tier} ({model_ok/n_tier:.1%})")
        print(f"    50/50:    {blend_ok:>4d}/{n_tier} ({blend_ok/n_tier:.1%})")
        print(f"    Adaptive: {adapt_ok:>4d}/{n_tier} ({adapt_ok/n_tier:.1%})")

    # When model and odds DISAGREE — who's right?
    print(f"\n{'='*75}")
    print(f"  WHEN MODEL AND ODDS DISAGREE")
    print(f"{'='*75}")
    disagree = ((model_p >= 0.5) != (odds_p >= 0.5))
    n_dis = disagree.sum()
    if n_dis > 0:
        odds_right = ((odds_p[disagree] >= 0.5).astype(int) == hw[disagree]).sum()
        model_right = ((model_p[disagree] >= 0.5).astype(int) == hw[disagree]).sum()

        print(f"\n  Total disagreements: {n_dis}/{len(hw)} ({n_dis/len(hw):.1%})")
        print(f"  Odds was right:  {odds_right}/{n_dis} ({odds_right/n_dis:.1%})")
        print(f"  Model was right: {model_right}/{n_dis} ({model_right/n_dis:.1%})")

        # By confidence of disagreement
        for label, lo, hi in [("Slight (<5%)", 0, 0.05),
                               ("Moderate (5-10%)", 0.05, 0.10),
                               ("Strong (>10%)", 0.10, 1.0)]:
            diff = np.abs(model_p - odds_p)
            sub_mask = disagree & (diff >= lo) & (diff < hi)
            n_sub = sub_mask.sum()
            if n_sub > 10:
                o_r = ((odds_p[sub_mask] >= 0.5).astype(int) == hw[sub_mask]).sum()
                m_r = ((model_p[sub_mask] >= 0.5).astype(int) == hw[sub_mask]).sum()
                print(f"    {label}: {n_sub} games — odds {o_r/n_sub:.1%}, model {m_r/n_sub:.1%}")


def main():
    args = parse_args()

    print("=" * 75)
    print("  COMPREHENSIVE STRATEGY BACKTEST")
    print("  Walk-forward: train on prior seasons, predict test season")
    print("=" * 75)

    # Load data
    print("\n  Loading data...")
    matches, ladders, odds = load_historical_data()

    # Elo params
    from predict_round import get_elo_params
    elo_params = get_elo_params(matches)

    # Build features for ALL matches
    print("\n  Building features for all matches...")
    all_data, feature_cols = build_all_features(matches, ladders, odds, elo_params)
    print(f"  Total matches with features: {len(all_data)}")
    print(f"  Years: {sorted(all_data['year'].unique())}")

    # Walk-forward predictions
    print(f"\n  Walk-forward prediction (test years: {args.test_years})...")
    predictions = walk_forward_predictions(
        all_data, feature_cols, args.test_years, args.min_train_year
    )

    # Save predictions for analysis
    out_path = PROJECT_ROOT / "outputs" / "reports" / "walkforward_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)
    print(f"\n  Saved {len(predictions)} predictions to {out_path}")

    # Evaluate strategies
    strategies, baseline, has_odds = evaluate_strategies(predictions)

    # Per-year breakdown
    per_year_breakdown(predictions)

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
