#!/usr/bin/env python3
"""
Backtest: Late Odds Refresh Strategy
=====================================
Tests whether re-checking odds before kickoff (closing line) improves
tipping accuracy vs using early-week odds (opening line).

Key question: for close games, does following the late market beat
sticking with our Tuesday tip?

Strategies tested:
  A. BASELINE:    Always tip opening-odds favourite (Tuesday)
  B. ALWAYS_CLOSE: Always tip closing-odds favourite (kickoff)
  C. BLEND_OPEN:  Our model blend (50/50 CatBoost+odds) with opening odds
  D. BLEND_CLOSE: Our model blend (50/50 CatBoost+odds) with closing odds
  E. SELECTIVE:   Use opening odds for LOCKs, closing for close games
  F. DRIFT:       Only flip when odds DRIFT across 50% (favourite changes)

Each strategy also tested with varying "close game" thresholds.

Usage:
    python scripts/backtest_late_odds.py
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


def main():
    t0 = time.time()

    print("=" * 70)
    print("  BACKTEST: Late Odds Refresh Strategy")
    print("=" * 70)

    # Load data
    odds = pd.read_parquet(PROCESSED_DIR / "odds.parquet")
    odds["date"] = pd.to_datetime(odds["date"])
    odds["year"] = odds["date"].dt.year

    # Filter to matches with both open AND close odds
    df = odds[
        odds["h2h_home_open"].notna() &
        odds["h2h_away_open"].notna() &
        odds["h2h_home_close"].notna() &
        odds["h2h_away_close"].notna() &
        odds["home_score"].notna() &
        odds["away_score"].notna()
    ].copy()

    # Remove draws
    df = df[df["home_score"] != df["away_score"]].copy()

    # Compute probabilities
    df["open_prob"] = (1 / df["h2h_home_open"]) / (1 / df["h2h_home_open"] + 1 / df["h2h_away_open"])
    df["close_prob"] = (1 / df["h2h_home_close"]) / (1 / df["h2h_home_close"] + 1 / df["h2h_away_close"])
    df["home_won"] = (df["home_score"] > df["away_score"]).astype(int)
    df["movement"] = df["close_prob"] - df["open_prob"]
    df["open_confidence"] = (df["open_prob"] - 0.5).abs() * 2  # 0=coin flip, 1=certain

    # Also compute "mid" odds (h2h_home/away = consensus/average line)
    df["mid_prob"] = (1 / df["h2h_home"]) / (1 / df["h2h_home"] + 1 / df["h2h_away"])

    test_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    df = df[df["year"].isin(test_years)].copy()

    n = len(df)
    print(f"\n  Dataset: {n} matches ({test_years[0]}-{test_years[-1]})")
    print(f"  Open-to-close movement >5%: {(df['movement'].abs() > 0.05).sum()} ({(df['movement'].abs() > 0.05).mean()*100:.1f}%)")
    print(f"  Favourite flips: {(np.sign(df['open_prob']-0.5) != np.sign(df['close_prob']-0.5)).sum()}")

    # ── Strategy A: Always open favourite (Tuesday tip) ──────────
    open_tip = df["open_prob"] >= 0.5
    open_correct = (open_tip == df["home_won"]).sum()

    # ── Strategy B: Always close favourite (kickoff) ─────────────
    close_tip = df["close_prob"] >= 0.5
    close_correct = (close_tip == df["home_won"]).sum()

    # ── Strategy C: tip mid-line favourite ────────────────────────
    mid_tip = df["mid_prob"] >= 0.5
    mid_correct = (mid_tip == df["home_won"]).sum()

    print(f"\n  {'Strategy':<45s} {'Correct':>8s} {'Acc':>7s} {'Δ':>5s}")
    print(f"  {'─'*70}")
    print(f"  {'A. Open-odds favourite (Tuesday)':<45s} {open_correct:>8d} {open_correct/n:>7.1%} {'base':>5s}")
    print(f"  {'B. Close-odds favourite (kickoff)':<45s} {close_correct:>8d} {close_correct/n:>7.1%} {close_correct-open_correct:>+5d}")
    print(f"  {'C. Mid-line favourite':<45s} {mid_correct:>8d} {mid_correct/n:>7.1%} {mid_correct-open_correct:>+5d}")

    # ── Strategy D: Selective — open for LOCKs, close for close games ──
    print(f"\n  {'Strategy D: Selective refresh':<45s}")
    print(f"  {'  (open for LOCKs, close for close games)':<45s}")
    print(f"  {'  threshold':<15s} {'correct':>8s} {'acc':>7s} {'Δ':>5s} {'flips':>6s} {'flip_acc':>8s}")
    print(f"  {'  ─'*22}")

    for thresh in [0.52, 0.53, 0.54, 0.55, 0.57, 0.60, 0.65]:
        is_close = df["open_prob"].between(1 - thresh, thresh)
        sel_tip = np.where(is_close, close_tip, open_tip)
        sel_correct = (sel_tip == df["home_won"].values).sum()
        # How many flipped?
        flipped = (sel_tip != open_tip.values).sum()
        flip_mask = sel_tip != open_tip.values
        if flipped > 0:
            flip_correct = ((sel_tip == df["home_won"].values) & flip_mask).sum()
            flip_acc = flip_correct / flipped
        else:
            flip_acc = 0
        close_n = is_close.sum()
        label = f"  close < {thresh:.0%} ({close_n} games)"
        print(f"  {label:<15s} {sel_correct:>8d} {sel_correct/n:>7.1%} "
              f"{sel_correct-open_correct:>+5d} {flipped:>6d} {flip_acc:>8.1%}")

    # ── Strategy E: Only flip when odds DRIFT across 50% ─────────
    drift_mask = (np.sign(df["open_prob"].values - 0.5) != np.sign(df["close_prob"].values - 0.5))
    drift_tip = np.where(drift_mask, close_tip.values, open_tip.values)
    drift_correct = (drift_tip == df["home_won"].values).sum()

    n_drifts = drift_mask.sum()
    if n_drifts > 0:
        drift_flips_correct = ((drift_tip == df["home_won"].values) & drift_mask).sum()
        drift_flip_acc = drift_flips_correct / n_drifts
    else:
        drift_flip_acc = 0

    print(f"\n  {'E. Drift-only (flip when fav changes)':<45s} {drift_correct:>8d} {drift_correct/n:>7.1%} "
          f"{drift_correct-open_correct:>+5d}  ({n_drifts} flips, {drift_flip_acc:.1%} flip acc)")

    # ── Per-year breakdown for best strategies ────────────────────
    print(f"\n  Per-year breakdown:")
    print(f"  {'Year':>6s} {'N':>5s} {'Open':>6s} {'Close':>6s} {'Drift':>6s} "
          f"{'Sel55':>6s} {'Δ_close':>7s} {'Δ_drift':>7s} {'Δ_sel55':>7s}")
    print(f"  {'─'*60}")

    for yr in sorted(df["year"].unique()):
        ym = df[df["year"] == yr]
        yn = len(ym)
        hw = ym["home_won"].values

        yo = (ym["open_prob"].values >= 0.5) == hw
        yc = (ym["close_prob"].values >= 0.5) == hw

        # Drift
        ydm = np.sign(ym["open_prob"].values - 0.5) != np.sign(ym["close_prob"].values - 0.5)
        ydt = np.where(ydm, ym["close_prob"].values >= 0.5, ym["open_prob"].values >= 0.5)
        yd = (ydt == hw).sum()

        # Selective 55%
        yis_close = ym["open_prob"].between(0.45, 0.55)
        yst = np.where(yis_close, ym["close_prob"].values >= 0.5, ym["open_prob"].values >= 0.5)
        ys = (yst == hw).sum()

        print(f"  {yr:>6d} {yn:>5d} {yo.sum():>6d} {yc.sum():>6d} {yd:>6d} "
              f"{ys:>6d} {yc.sum()-yo.sum():>+7d} {yd-yo.sum():>+7d} {ys-yo.sum():>+7d}")

    # ── Deep dive: close games only ──────────────────────────────
    print(f"\n  ── Close Games Deep Dive (open prob 45-55%) ──")
    close_df = df[df["open_prob"].between(0.45, 0.55)].copy()
    cn = len(close_df)
    chw = close_df["home_won"].values

    co = (close_df["open_prob"].values >= 0.5) == chw
    cc = (close_df["close_prob"].values >= 0.5) == chw
    cm = (close_df["mid_prob"].values >= 0.5) == chw

    print(f"  Matches: {cn}")
    print(f"  Open favourite:   {co.sum()}/{cn} ({co.mean():.1%})")
    print(f"  Close favourite:  {cc.sum()}/{cn} ({cc.mean():.1%}) ({cc.sum()-co.sum():+d})")
    print(f"  Mid favourite:    {cm.sum()}/{cn} ({cm.mean():.1%}) ({cm.sum()-co.sum():+d})")

    # Movement in close games
    close_movement = close_df["movement"].abs()
    print(f"\n  Movement in close games:")
    print(f"    Mean: {close_movement.mean():.3f}")
    print(f"    >5%:  {(close_movement > 0.05).sum()} ({(close_movement > 0.05).mean()*100:.1f}%)")
    print(f"    >10%: {(close_movement > 0.10).sum()} ({(close_movement > 0.10).mean()*100:.1f}%)")

    # When odds move >5% in close games, does following the movement help?
    big_move = close_df[close_movement > 0.05].copy()
    if len(big_move) > 0:
        bm_open = (big_move["open_prob"].values >= 0.5) == big_move["home_won"].values
        bm_close = (big_move["close_prob"].values >= 0.5) == big_move["home_won"].values
        print(f"\n  Close games with >5% movement ({len(big_move)} matches):")
        print(f"    Open favourite acc:  {bm_open.mean():.1%}")
        print(f"    Close favourite acc: {bm_close.mean():.1%} ({bm_close.sum()-bm_open.sum():+d})")

    # ── Is the DIRECTION of movement predictive? ──────────────────
    print(f"\n  ── Movement Direction Analysis ──")
    for label, subset in [("All matches", df), ("Close games (45-55%)", close_df)]:
        moved = subset[subset["movement"].abs() > 0.02].copy()
        if len(moved) == 0:
            continue
        # Did the market move toward the actual winner?
        moved_toward_winner = (
            ((moved["movement"] > 0) & (moved["home_won"] == 1)) |
            ((moved["movement"] < 0) & (moved["home_won"] == 0))
        ).sum()
        print(f"  {label} ({len(moved)} with >2% move):")
        print(f"    Market moved toward winner: {moved_toward_winner}/{len(moved)} ({moved_toward_winner/len(moved):.1%})")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
