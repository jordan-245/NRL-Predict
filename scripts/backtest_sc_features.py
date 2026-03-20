"""
A/B Backtest: SuperCoach matchup features impact on V4 pipeline.

Runs the V4 walk-forward backtest twice:
  A) WITHOUT SC matchup features (baseline)
  B) WITH SC matchup features

Compares accuracy, log loss, and Brier score across all model variants.

Usage:
    python scripts/backtest_sc_features.py
"""

from __future__ import annotations

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

from pipelines import v3, v4


SC_FEATURE_COLS = [
    "home_opp_sc_spine", "away_opp_sc_spine", "sc_spine_diff",
    "home_opp_sc_forward", "away_opp_sc_forward", "sc_forward_diff",
    "home_opp_sc_back", "away_opp_sc_back", "sc_back_diff",
    "home_opp_sc_total", "away_opp_sc_total", "sc_total_diff",
]


def build_dataset():
    """Build full V4 dataset including SC features (shared for both runs)."""
    print("=" * 80)
    print("  BUILDING DATASET (shared for A/B comparison)")
    print("=" * 80)

    matches, ladders, odds = v3.load_and_fix_homeaway()
    matches = v3.link_odds(matches, odds)
    elo_params = v3.tune_elo(matches, n_trials=50)
    matches = v3.backfill_elo(matches, elo_params)

    # V3 features
    matches = v3.compute_rolling_form_features(matches)
    matches = v3.compute_h2h_features(matches)
    matches = v3.compute_ladder_features(matches, ladders)
    matches = v3.compute_venue_features(matches)
    matches = v3.compute_odds_features(matches)
    matches = v3.compute_schedule_features(matches)
    matches = v3.compute_contextual_features(matches)
    matches = v3.compute_engineered_features(matches)

    # V4 features
    matches = v4.compute_v4_odds_features(matches)
    matches = v4.compute_scoring_consistency_features(matches)
    matches = v4.compute_attendance_features(matches)
    matches = v4.compute_kickoff_features(matches)
    matches = v4.compute_lineup_stability_features(matches)
    matches = v4.compute_player_impact_features(matches)
    matches = v4.compute_v4_engineered_features(matches)
    matches = v4.compute_sc_matchup_features(matches)

    features, feature_cols = v4.build_v4_feature_matrix(matches)
    return features, feature_cols


def run_backtest(features, feature_cols, label):
    """Run walk-forward + blend/stack and return results dict."""
    print(f"\n{'=' * 80}")
    print(f"  BACKTEST: {label}")
    print(f"  Features: {len(feature_cols)}")
    print(f"{'=' * 80}")

    all_results, model_oof, y_parts, odds_parts, _ = v4.walk_forward_backtest_v4(
        features, feature_cols
    )
    all_results = v4.v4_blend_and_stack(all_results, model_oof, y_parts, odds_parts)
    return all_results


def compare(results_a, results_b):
    """Print side-by-side comparison."""
    print("\n" + "=" * 80)
    print("  A/B COMPARISON: WITHOUT vs WITH SuperCoach Matchup Features")
    print("=" * 80)

    # Get common models
    common = sorted(set(results_a.keys()) & set(results_b.keys()))

    rows = []
    for model in common:
        a = results_a[model]
        b = results_b[model]
        rows.append({
            "Model": model,
            "Acc_A": a["accuracy"],
            "Acc_B": b["accuracy"],
            "Acc_Δ": b["accuracy"] - a["accuracy"],
            "LL_A": a["log_loss"],
            "LL_B": b["log_loss"],
            "LL_Δ": b["log_loss"] - a["log_loss"],
            "Brier_A": a.get("brier", np.nan),
            "Brier_B": b.get("brier", np.nan),
            "Brier_Δ": b.get("brier", np.nan) - a.get("brier", np.nan),
        })

    df = pd.DataFrame(rows).sort_values("LL_Δ", ascending=True)

    print(f"\n{'Model':<55} | {'Acc(A)':>7} {'Acc(B)':>7} {'Δ':>7} | "
          f"{'LL(A)':>7} {'LL(B)':>7} {'Δ':>7} | "
          f"{'Brier(A)':>8} {'Brier(B)':>8} {'Δ':>7}")
    print("-" * 140)

    improved = 0
    degraded = 0
    for _, r in df.iterrows():
        marker = ""
        if r["LL_Δ"] < -0.0005:
            marker = " ✓"
            improved += 1
        elif r["LL_Δ"] > 0.0005:
            marker = " ✗"
            degraded += 1

        print(f"{r['Model']:<55} | {r['Acc_A']:7.4f} {r['Acc_B']:7.4f} {r['Acc_Δ']:+7.4f} | "
              f"{r['LL_A']:7.4f} {r['LL_B']:7.4f} {r['LL_Δ']:+7.4f} | "
              f"{r['Brier_A']:8.4f} {r['Brier_B']:8.4f} {r['Brier_Δ']:+7.4f}{marker}")

    print("-" * 140)
    print(f"\n  Summary: {improved} models improved, {degraded} degraded, "
          f"{len(df) - improved - degraded} neutral (±0.0005 LL)")

    # Best model comparison
    best_a_model = min(results_a, key=lambda k: results_a[k]["log_loss"])
    best_b_model = min(results_b, key=lambda k: results_b[k]["log_loss"])
    best_a = results_a[best_a_model]
    best_b = results_b[best_b_model]

    print(f"\n  Best WITHOUT SC: {best_a_model}")
    print(f"    Acc={best_a['accuracy']:.4f}  LL={best_a['log_loss']:.4f}")
    print(f"  Best WITH SC:    {best_b_model}")
    print(f"    Acc={best_b['accuracy']:.4f}  LL={best_b['log_loss']:.4f}")
    print(f"  Δ LL: {best_b['log_loss'] - best_a['log_loss']:+.4f}")

    if best_b['log_loss'] < best_a['log_loss']:
        pct = (best_a['log_loss'] - best_b['log_loss']) / best_a['log_loss'] * 100
        print(f"\n  >>> SC FEATURES IMPROVE BEST MODEL by {pct:.3f}% <<<")
    else:
        print(f"\n  SC features did not improve the best model.")

    # Save comparison
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / "sc_features_ab_comparison.csv", index=False)
    print(f"\n  Saved: {report_dir / 'sc_features_ab_comparison.csv'}")

    return df


def main():
    start = time.time()

    # Build dataset once (includes SC features)
    features_full, feature_cols_full = build_dataset()

    # Run A: WITHOUT SC features
    cols_no_sc = [c for c in feature_cols_full if c not in SC_FEATURE_COLS]
    features_no_sc = features_full.copy()
    results_a = run_backtest(features_no_sc, cols_no_sc, "A: WITHOUT SC matchup features")

    # Run B: WITH SC features
    results_b = run_backtest(features_full, feature_cols_full, "B: WITH SC matchup features")

    # Compare
    comp_df = compare(results_a, results_b)

    elapsed = time.time() - start
    print(f"\n  Total A/B backtest time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
