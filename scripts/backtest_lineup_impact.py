#!/usr/bin/env python3
"""
Backtest: Player Impact Lineup Adjustments
==========================================
Measures whether adjusting tip probabilities based on player impact
scores (when starters are scratched) improves tipping accuracy.

Walk-forward design (no look-ahead):
  For each test season Y (2020-2025):
    1. Compute player impact scores using data from years [Y-3, Y-1] only
    2. For each match in season Y:
       a. Get "expected starters" from that team's last 5 matches
       b. Get "actual starters" from who played
       c. Compute lineup diff → raw impact adjustment per team
    3. Grid search over capping/thresholding parameters

Phase 1: precompute per-match raw adjustments (slow, ~2min)
Phase 2: grid search over caps/thresholds (fast, pure numpy)

Usage:
    python scripts/backtest_lineup_impact.py
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


# ──────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────

def load_all_data():
    apps = pd.read_parquet(PROCESSED_DIR / "player_appearances.parquet")
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    odds = pd.read_parquet(PROCESSED_DIR / "odds.parquet")
    odds["date"] = pd.to_datetime(odds["date"])
    odds["year"] = odds["date"].dt.year
    return apps, matches, odds


# ──────────────────────────────────────────────────────────────────
# Walk-forward impact scores (precomputed per year)
# ──────────────────────────────────────────────────────────────────

def compute_impact_for_year(
    apps: pd.DataFrame,
    matches: pd.DataFrame,
    target_year: int,
    window: int = 3,
    min_starts: int = 10,
    min_absences: int = 3,
) -> pd.DataFrame:
    """Compute impact scores using only [target_year-window, target_year-1]."""
    from processing.player_impact import build_team_match_log, SPINE_POSITIONS

    ws = target_year - window
    we = target_year - 1

    hist_m = matches[(matches["year"] >= ws) & (matches["year"] <= we)].copy()
    hist_a = apps[(apps["year"] >= ws) & (apps["year"] <= we)].copy()

    if hist_m.empty or hist_a.empty:
        return pd.DataFrame()

    team_log = build_team_match_log(hist_m)
    starters = hist_a[hist_a["is_starter"]].copy()
    starter_set = set(zip(starters["match_id"], starters["team"], starters["player_id"]))

    pt = starters.groupby(["player_id", "team"]).size().reset_index(name="count")

    results = []
    for _, row in pt.iterrows():
        pid, team = row["player_id"], row["team"]

        tm = team_log[(team_log["team"] == team) & (team_log["year"] >= ws) & (team_log["year"] <= we)]
        if len(tm) < min_starts + min_absences:
            continue

        tm = tm.copy()
        tm["ps"] = tm["match_id"].apply(lambda mid: (mid, team, pid) in starter_set)
        wp = tm[tm["ps"]]
        wo = tm[~tm["ps"]]

        if len(wp) < min_starts or len(wo) < min_absences:
            continue

        r_with = (wp["result"] - wp["elo_expected"]).mean()
        r_without = (wo["result"] - wo["elo_expected"]).mean()
        elo_impact = r_with - r_without

        mg = min(len(wp), len(wo))
        tg = len(wp) + len(wo)
        conf = min(1.0, mg / 15.0) * min(1.0, tg / 40.0)

        pa = starters[(starters["player_id"] == pid) & (starters["team"] == team)]
        if pa.empty:
            continue

        pname = pa["player_name"].mode().iloc[0] if not pa["player_name"].mode().empty else str(pid)
        pos = pa["position"].mode().iloc[0] if not pa["position"].mode().empty else "UNK"

        results.append({
            "player_id": pid, "player_name": pname, "team": team,
            "position": pos, "is_spine": pos in SPINE_POSITIONS,
            "elo_adj_impact": round(elo_impact, 4),
            "confidence": round(conf, 3),
            "weighted_impact": round(elo_impact * conf, 4),
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────
# Precompute per-match raw adjustments
# ──────────────────────────────────────────────────────────────────

def precompute_match_data(
    apps: pd.DataFrame,
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    test_years: list[int],
) -> pd.DataFrame:
    """Precompute everything needed per match for fast grid search.

    Returns DataFrame with one row per match:
        year, home, away, home_won, base_prob,
        home_adj_weighted, away_adj_weighted,
        home_adj_elo, away_adj_elo,
        home_adj_weighted_cf03, away_adj_weighted_cf03, ... (confidence-filtered variants)
    """
    print("  Phase 1: Precomputing per-match adjustments...")

    # Precompute impact scores for each test year
    impact_cache = {}
    for year in test_years:
        t0 = time.time()
        for ms, ma in [(5, 2), (10, 3), (15, 5)]:
            key = (year, ms, ma)
            impact_cache[key] = compute_impact_for_year(apps, matches, year, 3, ms, ma)
        print(f"    {year}: impact scores computed ({time.time()-t0:.1f}s)")

    # Build team → match appearance index for fast lookup
    starter_apps = apps[apps["is_starter"]].copy()
    starter_apps = starter_apps.sort_values(["year", "round"])

    # For expected starters: precompute per team+match
    print("    Building expected starters index...")
    # Group by team, get ordered match_ids
    team_match_order = {}
    for team, grp in starter_apps.groupby("team"):
        ordered_mids = list(grp.drop_duplicates("match_id")["match_id"].values)
        team_match_order[team] = ordered_mids

    # Build actual starters per match
    actual_starters = {}  # (match_id, team) → {jersey: surname}
    for (mid, team), grp in starter_apps.groupby(["match_id", "team"]):
        actual_starters[(mid, team)] = {
            int(r["jersey_number"]): r["player_name"]
            for _, r in grp.iterrows()
            if r["jersey_number"] <= 13
        }

    # Build expected starters per match (from prior 5 matches of that team)
    print("    Building expected starters per match...")
    expected_starters = {}
    for team, mid_list in team_match_order.items():
        for i, mid in enumerate(mid_list):
            if i < 5:
                continue  # need at least 5 prior matches
            prior_mids = mid_list[max(0, i-5):i]
            # Get most common player at each jersey from prior matches
            prior_apps = starter_apps[
                (starter_apps["team"] == team) &
                (starter_apps["match_id"].isin(prior_mids))
            ]
            exp = {}
            for jersey in range(1, 14):
                pa = prior_apps[prior_apps["jersey_number"] == jersey]
                if not pa.empty:
                    mode = pa["player_name"].mode()
                    if not mode.empty:
                        exp[jersey] = mode.iloc[0]
            expected_starters[(mid, team)] = exp

    # Now compute raw adjustments for each match
    print("    Computing raw adjustments per match...")
    rows = []

    for year in test_years:
        year_matches = matches[
            (matches["year"] == year) & matches["home_score"].notna()
        ].copy()
        year_odds = odds[odds["year"] == year]

        for _, m in year_matches.iterrows():
            home, away = m["home_team"], m["away_team"]
            hs, as_ = m["home_score"], m["away_score"]
            if hs == as_:
                continue

            mid = f"{m['year']}_r{m['round']}_{home}_v_{away}"
            home_won = 1 if hs > as_ else 0

            # Get odds prob
            om = year_odds[(year_odds["home_team"] == home) & (year_odds["away_team"] == away)]
            swapped = False
            if om.empty:
                om = year_odds[(year_odds["home_team"] == away) & (year_odds["away_team"] == home)]
                swapped = True
            if om.empty:
                continue

            if swapped:
                h2h_h = om.iloc[0].get("h2h_away", None)
                h2h_a = om.iloc[0].get("h2h_home", None)
            else:
                h2h_h = om.iloc[0].get("h2h_home", None)
                h2h_a = om.iloc[0].get("h2h_away", None)

            if pd.isna(h2h_h) or pd.isna(h2h_a) or h2h_h <= 0 or h2h_a <= 0:
                continue

            base_prob = (1 / h2h_h) / (1 / h2h_h + 1 / h2h_a)

            # Get expected and actual starters
            home_exp = expected_starters.get((mid, home), {})
            home_act = actual_starters.get((mid, home), {})
            away_exp = expected_starters.get((mid, away), {})
            away_act = actual_starters.get((mid, away), {})

            if not home_exp or not home_act or not away_exp or not away_act:
                continue

            # Compute raw adjustments for each (min_starts, min_absences) combo
            row = {
                "year": year, "match_id": mid,
                "home": home, "away": away,
                "home_won": home_won, "base_prob": base_prob,
                "round": m["round"],
            }

            for (ms, ma) in [(5, 2), (10, 3), (15, 5)]:
                impact_df = impact_cache.get((year, ms, ma))
                if impact_df is None or impact_df.empty:
                    for suffix in [f"_s{ms}a{ma}_w", f"_s{ms}a{ma}_e",
                                   f"_s{ms}a{ma}_w_cf03", f"_s{ms}a{ma}_e_cf03",
                                   f"_s{ms}a{ma}_w_cf05", f"_s{ms}a{ma}_e_cf05"]:
                        row[f"home_adj{suffix}"] = 0.0
                        row[f"away_adj{suffix}"] = 0.0
                    continue

                for side, exp, act, team in [
                    ("home", home_exp, home_act, home),
                    ("away", away_exp, away_act, away),
                ]:
                    for icol, isuf in [("weighted_impact", "w"), ("elo_adj_impact", "e")]:
                        for cf, cfsuf in [(0.0, ""), (0.3, "_cf03"), (0.5, "_cf05")]:
                            adj = 0.0
                            for jersey, exp_name in exp.items():
                                act_name = act.get(jersey)
                                if act_name is None:
                                    continue
                                # Surname comparison
                                es = exp_name.split()[-1].lower() if exp_name else ""
                                as_ = act_name.split()[-1].lower() if act_name else ""
                                if es == as_:
                                    continue
                                # Look up impact
                                mask = (impact_df["team"] == team) & \
                                       impact_df["player_name"].str.lower().str.endswith(es)
                                pr = impact_df[mask]
                                if pr.empty:
                                    continue
                                imp = float(pr.iloc[0][icol])
                                c = float(pr.iloc[0].get("confidence", 1.0))
                                if c < cf:
                                    continue
                                adj -= imp
                            key = f"{side}_adj_s{ms}a{ma}_{isuf}{cfsuf}"
                            row[key] = adj

            rows.append(row)

        print(f"    {year}: {sum(1 for r in rows if r['year']==year)} matches with adjustments")

    df = pd.DataFrame(rows)
    print(f"  Phase 1 complete: {len(df)} total matches\n")
    return df


# ──────────────────────────────────────────────────────────────────
# Phase 2: fast grid search
# ──────────────────────────────────────────────────────────────────

def grid_search(match_df: pd.DataFrame) -> pd.DataFrame:
    """Fast grid search over capping/threshold parameters."""
    print("  Phase 2: Grid search...")

    max_net_adjs = [0.03, 0.05, 0.075, 0.10, 0.125, 0.15]
    swing_thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    # Build list of adjustment column sets
    adj_configs = []
    for ms, ma in [(5, 2), (10, 3), (15, 5)]:
        for icol, isuf in [("weighted_impact", "w"), ("elo_adj_impact", "e")]:
            for cf, cfsuf in [(0.0, ""), (0.3, "_cf03"), (0.5, "_cf05")]:
                hcol = f"home_adj_s{ms}a{ma}_{isuf}{cfsuf}"
                acol = f"away_adj_s{ms}a{ma}_{isuf}{cfsuf}"
                if hcol in match_df.columns:
                    adj_configs.append({
                        "min_starts": ms, "min_absences": ma,
                        "impact_col": icol, "conf_floor": cf,
                        "home_col": hcol, "away_col": acol,
                    })

    base_prob = match_df["base_prob"].values
    home_won = match_df["home_won"].values
    base_tip_home = base_prob >= 0.5
    base_correct = (base_tip_home == home_won).sum()
    n = len(match_df)

    results = []

    for cfg in adj_configs:
        home_raw = match_df[cfg["home_col"]].values
        away_raw = match_df[cfg["away_col"]].values

        for max_net in max_net_adjs:
            # Cap per-team at 0.15, then net
            h_capped = np.clip(home_raw, -0.15, 0.15)
            a_capped = np.clip(away_raw, -0.15, 0.15)
            total_adj = np.clip(h_capped - a_capped, -max_net, max_net)

            adj_prob = np.clip(base_prob + total_adj, 0.05, 0.95)

            for swing_t in swing_thresholds:
                adj_tip_home = adj_prob >= 0.5
                old_conf = np.abs(base_prob - 0.5) * 2

                # Block swings above threshold
                would_swing = adj_tip_home != base_tip_home
                blocked = would_swing & (old_conf > swing_t)
                adj_tip_home = np.where(blocked, base_tip_home, adj_tip_home)

                adj_correct = (adj_tip_home == home_won).sum()
                delta = int(adj_correct - base_correct)

                # Swing stats
                actual_swings = would_swing & ~blocked
                n_swings = actual_swings.sum()
                swing_ok = (actual_swings & (adj_tip_home == home_won)).sum()
                swing_bad = (actual_swings & (adj_tip_home != home_won)).sum()

                # Direction: did adjustment move prob toward correct answer?
                has_adj = np.abs(total_adj) > 0.001
                correct_dir = has_adj & (
                    ((total_adj > 0) & (home_won == 1)) |
                    ((total_adj < 0) & (home_won == 0))
                )
                n_with_adj = has_adj.sum()
                n_correct_dir = correct_dir.sum()

                results.append({
                    "max_net_adj": max_net,
                    "impact_col": cfg["impact_col"],
                    "min_starts": cfg["min_starts"],
                    "min_absences": cfg["min_absences"],
                    "conf_floor": cfg["conf_floor"],
                    "swing_threshold": swing_t,
                    "matches": n,
                    "base_correct": int(base_correct),
                    "adj_correct": int(adj_correct),
                    "base_acc": base_correct / n,
                    "adj_acc": adj_correct / n,
                    "delta": delta,
                    "swings": int(n_swings),
                    "swing_correct": int(swing_ok),
                    "swing_wrong": int(swing_bad),
                    "swing_acc": swing_ok / n_swings if n_swings else 0,
                    "matches_with_adj": int(n_with_adj),
                    "direction_correct": int(n_correct_dir),
                    "direction_rate": n_correct_dir / n_with_adj if n_with_adj else 0,
                })

    return pd.DataFrame(results)


def main():
    t0 = time.time()

    print("=" * 70)
    print("  BACKTEST: Player Impact Lineup Adjustments")
    print("=" * 70)

    print("\n  Loading data...")
    apps, matches, odds = load_all_data()

    test_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Phase 1: precompute
    match_df = precompute_match_data(apps, matches, odds, test_years)

    # Phase 2: grid search
    results_df = grid_search(match_df)

    # Save
    output_path = PROJECT_ROOT / "outputs" / "reports" / "lineup_impact_backtest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    elapsed = time.time() - t0
    n_combos = len(results_df)
    print(f"\n  Completed {n_combos} combinations in {elapsed:.0f}s")
    print(f"  Results saved to {output_path}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    base = results_df.iloc[0]
    print(f"\n  Baseline (odds favourite, no lineup adj):")
    print(f"    Matches: {int(base['matches'])}")
    print(f"    Correct: {int(base['base_correct'])} ({base['base_acc']:.1%})")

    # Best
    best = results_df.sort_values("delta", ascending=False).iloc[0]
    print(f"\n  ┌─ Best configuration ─────────────────────────────")
    print(f"  │  max_net_adj:      {best['max_net_adj']}")
    print(f"  │  impact_col:       {best['impact_col']}")
    print(f"  │  min_starts:       {int(best['min_starts'])}")
    print(f"  │  min_absences:     {int(best['min_absences'])}")
    print(f"  │  conf_floor:       {best['conf_floor']}")
    print(f"  │  swing_threshold:  {best['swing_threshold']}")
    print(f"  │  ───────────────────────────────────────────────")
    print(f"  │  Accuracy:         {best['adj_acc']:.1%} (base: {best['base_acc']:.1%})")
    print(f"  │  Delta:            {int(best['delta']):+d} tips / {int(best['matches'])} matches")
    print(f"  │  Swings:           {int(best['swings'])} total "
          f"({int(best['swing_correct'])} ✓, {int(best['swing_wrong'])} ✗)")
    if best['swings'] > 0:
        print(f"  │  Swing accuracy:   {best['swing_acc']:.1%}")
    print(f"  │  Direction rate:   {best['direction_rate']:.1%}")
    print(f"  └────────────────────────────────────────────────────")

    # Worst
    worst = results_df.sort_values("delta", ascending=True).iloc[0]
    print(f"\n  Worst: delta={int(worst['delta']):+d}, max_net={worst['max_net_adj']}, "
          f"swing_t={worst['swing_threshold']}, {worst['impact_col']}")

    # Distribution
    pos = (results_df["delta"] > 0).sum()
    zero = (results_df["delta"] == 0).sum()
    neg = (results_df["delta"] < 0).sum()
    print(f"\n  Distribution across {n_combos} configs:")
    print(f"    Positive: {pos:>4d} ({pos/n_combos*100:.0f}%)")
    print(f"    Zero:     {zero:>4d} ({zero/n_combos*100:.0f}%)")
    print(f"    Negative: {neg:>4d} ({neg/n_combos*100:.0f}%)")

    # Top 15
    print(f"\n  Top 15:")
    print(f"  {'adj':>5s} {'impact':>10s} {'st':>3s} {'ab':>3s} {'cf':>4s} "
          f"{'sw_t':>5s} {'Δ':>4s} {'acc':>6s} {'swings':>6s} {'sw%':>5s} {'dir%':>5s}")
    print(f"  {'─'*66}")
    for _, r in results_df.sort_values("delta", ascending=False).head(15).iterrows():
        ic = "weighted" if "weighted" in r["impact_col"] else "elo_adj"
        print(f"  {r['max_net_adj']:>5.3f} {ic:>10s} {int(r['min_starts']):>3d} "
              f"{int(r['min_absences']):>3d} {r['conf_floor']:>4.1f} "
              f"{r['swing_threshold']:>5.2f} {int(r['delta']):>+4d} "
              f"{r['adj_acc']:>6.1%} {int(r['swings']):>6d} "
              f"{r['swing_acc']:>5.1%} {r['direction_rate']:>5.1%}")

    # Per-year breakdown for best config
    print(f"\n  Per-year breakdown (best config):")
    # Re-run best config per year
    best_hcol = f"home_adj_s{int(best['min_starts'])}a{int(best['min_absences'])}_"
    best_hcol += "w" if "weighted" in best["impact_col"] else "e"
    cf_map = {0.0: "", 0.3: "_cf03", 0.5: "_cf05"}
    best_hcol += cf_map.get(best["conf_floor"], "")
    best_acol = best_hcol.replace("home_adj", "away_adj")

    if best_hcol in match_df.columns:
        for yr in sorted(match_df["year"].unique()):
            ym = match_df[match_df["year"] == yr]
            bp = ym["base_prob"].values
            hw = ym["home_won"].values
            hr = np.clip(ym[best_hcol].values, -0.15, 0.15)
            ar = np.clip(ym[best_acol].values, -0.15, 0.15)
            ta = np.clip(hr - ar, -best["max_net_adj"], best["max_net_adj"])
            ap = np.clip(bp + ta, 0.05, 0.95)
            bt = bp >= 0.5
            at = ap >= 0.5
            oc = np.abs(bp - 0.5) * 2
            ws = (at != bt) & (oc > best["swing_threshold"])
            at = np.where(ws, bt, at)
            bc = (bt == hw).sum()
            ac = (at == hw).sum()
            d = ac - bc
            print(f"    {int(yr)}: base={bc}/{len(ym)} ({bc/len(ym):.1%}), "
                  f"adj={ac}/{len(ym)} ({ac/len(ym):.1%}), delta={d:+d}")

    print()


if __name__ == "__main__":
    main()
