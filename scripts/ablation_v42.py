"""
V4.2 Feature Ablation — Test each new source individually.

Runs 3 configurations (weather-only baseline already recorded):
  A) Weather + Odds Movement (no workload)
  B) Weather + Workload (no odds movement)
  C) All 4 sources combined

Outputs comparison table.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pipelines import v4, v3

# Baseline results (from prior runs)
BASELINE = {
    "V4.1 (no new features)": {"acc_opt": 0.6859, "ll_opt": 0.5898, "acc_wf": 0.6923, "ll_wf": 0.5900},
    "V4.2 weather only":      {"acc_opt": 0.7039, "ll_opt": 0.5639, "acc_wf": 0.7117, "ll_wf": 0.5733},
}

# Feature groups to ablate
WEATHER_FEATURES = [
    "temperature_c", "precipitation_mm", "wind_speed_kmh",
    "is_rainy", "is_windy", "is_cold_actual",
    "ground_not_good", "ground_severity",
    "rain_x_wind", "bad_conditions_score",
]

ODDS_MOVEMENT_FEATURES = [
    "prob_shift", "prob_shift_abs", "sharp_money_flag",
    "favourite_drift", "closing_range_pct", "line_overreaction",
]

WORKLOAD_FEATURES = [
    "home_starter_mins_avg_3", "away_starter_mins_avg_3", "workload_diff_3",
    "home_spine_mins_avg_3", "away_spine_mins_avg_3", "spine_workload_diff_3",
    "home_heavy_load_count", "away_heavy_load_count",
    "is_origin_period", "origin_round_number",
]


def run_ablation(features_df, all_feature_cols, exclude_features, label):
    """Run walk-forward backtest excluding specified features."""
    print(f"\n{'='*80}")
    print(f"  ABLATION: {label}")
    print(f"  Excluding: {len(exclude_features)} features")
    print(f"{'='*80}")

    # Filter feature columns
    active_cols = [c for c in all_feature_cols if c not in exclude_features]
    print(f"  Active features: {len(active_cols)} (was {len(all_feature_cols)})")

    # Run walk-forward backtest
    all_results, model_oof, y_parts, odds_parts, year_parts = v4.walk_forward_backtest_v4(
        features_df, active_cols
    )

    # Run blending
    all_results = v4.v4_blend_and_stack(all_results, model_oof, y_parts, odds_parts)

    # Extract key metrics
    metrics = {}
    for name, res in all_results.items():
        if "OptBlend V4-All9+Odds" in name and "Cal" not in name and "WF" not in name:
            metrics["acc_opt"] = res["accuracy"]
            metrics["ll_opt"] = res["log_loss"]
        if "WF-OptBlend WF-All9+Odds" in name:
            metrics["acc_wf"] = res["accuracy"]
            metrics["ll_wf"] = res["log_loss"]

    return metrics


def main():
    # Load features (already computed with all sources)
    features_path = v4.FEATURES_DIR / "features_v4.parquet"
    if not features_path.exists():
        print("ERROR: features_v4.parquet not found. Run pipelines/v4.py first.")
        return

    features = pd.read_parquet(features_path)

    # Get full feature column list
    _, all_feature_cols = v4.build_v4_feature_matrix(features)

    # Ensure all groups are present
    for group_name, group_feats in [("weather", WEATHER_FEATURES),
                                     ("odds_movement", ODDS_MOVEMENT_FEATURES),
                                     ("workload", WORKLOAD_FEATURES)]:
        present = [f for f in group_feats if f in all_feature_cols]
        print(f"  {group_name}: {len(present)}/{len(group_feats)} features present")

    results = dict(BASELINE)

    # A) Weather + Odds Movement (exclude workload)
    m = run_ablation(features, all_feature_cols, WORKLOAD_FEATURES, "Weather + Odds Movement")
    results["A) Weather + Odds"] = m

    # B) Weather + Workload (exclude odds movement)
    m = run_ablation(features, all_feature_cols, ODDS_MOVEMENT_FEATURES, "Weather + Workload")
    results["B) Weather + Workload"] = m

    # C) All combined (exclude nothing new)
    m = run_ablation(features, all_feature_cols, [], "All 4 Sources")
    results["C) All 4 Sources"] = m

    # Print comparison
    print("\n" + "=" * 100)
    print("  V4.2 FEATURE ABLATION RESULTS")
    print("=" * 100)
    print(f"  {'Config':35s} {'OptAcc':>8s} {'OptLL':>8s} {'WF-Acc':>8s} {'WF-LL':>8s} {'Δ Acc':>8s} {'Δ LL':>8s}")
    print("-" * 100)

    base_acc = BASELINE["V4.2 weather only"]["acc_wf"]
    base_ll = BASELINE["V4.2 weather only"]["ll_wf"]

    for name, m in results.items():
        d_acc = m.get("acc_wf", 0) - base_acc if "acc_wf" in m else 0
        d_ll = m.get("ll_wf", 0) - base_ll if "ll_wf" in m else 0
        marker = "✓" if d_acc > 0 and d_ll < 0 else ("≈" if abs(d_acc) < 0.002 else "✗")
        if name in BASELINE:
            marker = "—"
        print(f"  {name:35s} {m.get('acc_opt', 0):8.4f} {m.get('ll_opt', 0):8.4f} "
              f"{m.get('acc_wf', 0):8.4f} {m.get('ll_wf', 0):8.4f} "
              f"{d_acc:+8.4f} {d_ll:+8.4f} {marker}")

    print()
    print("  Δ is relative to V4.2 weather-only baseline (WF-OptBlend)")
    print("  ✓ = both metrics improve, ✗ = either metric worsens, ≈ = marginal")


if __name__ == "__main__":
    main()
