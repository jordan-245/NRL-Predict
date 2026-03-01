#!/usr/bin/env python3
"""
Weekly Evaluation & Learning Loop
==================================
After each round:
  1. Load predictions from the round
  2. Fetch actual results
  3. Score each prediction (correct/wrong)
  4. Track cumulative season performance
  5. Compare model vs odds vs blend accuracy
  6. Detect model drift (is model getting worse?)
  7. Send Telegram performance report
  8. Log detailed per-game results for analysis

Usage:
    python scripts/weekly_eval.py                    # auto-detect last round
    python scripts/weekly_eval.py --round 5          # specific round
    python scripts/weekly_eval.py --season-report    # full season summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Auto-load .env
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            import os
            os.environ.setdefault(_k.strip(), _v.strip())

from config.settings import PROCESSED_DIR
from config.team_mappings import standardise_team_name

PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
AEST = timezone(timedelta(hours=10))


def log(msg: str):
    ts = datetime.now(AEST).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ── Load predictions and results ─────────────────────────────────

def load_predictions(round_num: int, year: int) -> pd.DataFrame | None:
    """Load predictions CSV for a given round."""
    path = PREDICTIONS_DIR / f"round_{round_num}_{year}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_results(round_num: int, year: int) -> pd.DataFrame:
    """Load match results for a given round from matches.parquet."""
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    m = matches[(matches["year"] == year) & (matches["round"] == str(round_num))].copy()
    m = m[m["home_score"].notna()].copy()
    return m


def match_predictions_to_results(preds: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Join predictions with actual results."""
    rows = []
    for _, p in preds.iterrows():
        try:
            p_home = standardise_team_name(p["home_team"])
            p_away = standardise_team_name(p["away_team"])
        except KeyError:
            continue

        # Find matching result
        matched = False
        for _, r in results.iterrows():
            try:
                r_home = standardise_team_name(r["home_team"])
                r_away = standardise_team_name(r["away_team"])
            except KeyError:
                continue

            if (r_home == p_home and r_away == p_away) or \
               (r_home == p_away and r_away == p_home):
                home_score = r["home_score"]
                away_score = r["away_score"]

                # Determine actual winner
                if r_home == p_home:
                    actual_winner = p_home if home_score > away_score else p_away
                    actual_margin = abs(home_score - away_score)
                else:
                    actual_winner = p_away if home_score > away_score else p_home
                    actual_margin = abs(home_score - away_score)

                if home_score == away_score:
                    actual_winner = "DRAW"

                try:
                    tip = standardise_team_name(p["tip"])
                except KeyError:
                    tip = p["tip"]

                # Get odds favourite
                odds_prob = p.get("odds_home_prob", 0.5)
                odds_tip = p_home if odds_prob >= 0.5 else p_away

                # Get model prediction
                model_prob = p.get("model_CAT_top50", p["home_win_prob"])
                model_tip = p_home if model_prob >= 0.5 else p_away

                rows.append({
                    "home_team": p_home,
                    "away_team": p_away,
                    "tip": tip,
                    "odds_tip": odds_tip,
                    "model_tip": model_tip,
                    "actual_winner": actual_winner,
                    "actual_margin": actual_margin,
                    "home_win_prob": p["home_win_prob"],
                    "odds_home_prob": odds_prob,
                    "model_prob": model_prob,
                    "confidence": p.get("confidence", abs(p["home_win_prob"] - 0.5) * 2),
                    "home_score": home_score,
                    "away_score": away_score,
                    "tip_correct": 1 if tip == actual_winner else 0,
                    "odds_correct": 1 if odds_tip == actual_winner else 0,
                    "model_correct": 1 if model_tip == actual_winner else 0,
                })
                matched = True
                break

        if not matched:
            log(f"  ⚠ No result for {p['home_team']} v {p['away_team']}")

    return pd.DataFrame(rows)


# ── Season log ───────────────────────────────────────────────────

def season_log_path(year: int) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    return EVAL_DIR / f"season_{year}.csv"


def load_season_log(year: int) -> pd.DataFrame:
    path = season_log_path(year)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_round_results(year: int, round_num: int, matched: pd.DataFrame):
    """Append round results to the season log."""
    path = season_log_path(year)

    matched = matched.copy()
    matched["year"] = year
    matched["round"] = round_num

    if path.exists():
        existing = pd.read_csv(path)
        # Remove any previous entries for this round (re-evaluation)
        existing = existing[existing["round"] != round_num]
        combined = pd.concat([existing, matched], ignore_index=True)
    else:
        combined = matched

    combined.to_csv(path, index=False)
    log(f"  Saved {len(matched)} results to {path}")


# ── Performance metrics ──────────────────────────────────────────

def compute_metrics(season: pd.DataFrame) -> dict:
    """Compute season performance metrics."""
    n = len(season)
    if n == 0:
        return {}

    tip_correct = season["tip_correct"].sum()
    odds_correct = season["odds_correct"].sum()
    model_correct = season["model_correct"].sum()

    metrics = {
        "games": n,
        "rounds": season["round"].nunique(),
        "tip_correct": int(tip_correct),
        "tip_accuracy": tip_correct / n,
        "odds_correct": int(odds_correct),
        "odds_accuracy": odds_correct / n,
        "model_correct": int(model_correct),
        "model_accuracy": model_correct / n,
        "tip_vs_odds": int(tip_correct - odds_correct),
        "tip_vs_model": int(tip_correct - model_correct),
    }

    # By confidence tier
    for label, lo, hi in [("lock", 0.30, 1.0), ("lean", 0.10, 0.30), ("tossup", 0.0, 0.10)]:
        mask = season["confidence"].between(lo, hi)
        tier_n = mask.sum()
        if tier_n > 0:
            tier_correct = season.loc[mask, "tip_correct"].sum()
            tier_odds = season.loc[mask, "odds_correct"].sum()
            metrics[f"{label}_games"] = int(tier_n)
            metrics[f"{label}_correct"] = int(tier_correct)
            metrics[f"{label}_accuracy"] = tier_correct / tier_n
            metrics[f"{label}_odds_accuracy"] = tier_odds / tier_n
            metrics[f"{label}_edge"] = int(tier_correct - tier_odds)

    # Recent form (last 3 rounds)
    if season["round"].nunique() >= 3:
        last_3_rounds = sorted(season["round"].unique())[-3:]
        recent = season[season["round"].isin(last_3_rounds)]
        metrics["recent_3r_accuracy"] = recent["tip_correct"].mean()
        metrics["recent_3r_odds_accuracy"] = recent["odds_correct"].mean()

    return metrics


# ── Telegram report ──────────────────────────────────────────────

def send_round_report(round_num: int, matched: pd.DataFrame, metrics: dict, year: int):
    """Send round evaluation report to Telegram."""
    try:
        from scripts.telegram_notify import send_message, _esc
    except ImportError:
        log("  Cannot send Telegram (import failed)")
        return

    tip_ok = matched["tip_correct"].sum()
    odds_ok = matched["odds_correct"].sum()
    n = len(matched)

    lines = [
        f"📊 <b>Round {round_num} Evaluation</b>",
        "",
        f"<b>This Round:</b> {tip_ok}/{n} correct",
        f"  Odds favourite: {odds_ok}/{n}",
        f"  Model edge: {tip_ok - odds_ok:+d}",
        "",
    ]

    # Per-game breakdown
    for _, g in matched.iterrows():
        home_short = g["home_team"].split()[-1]
        away_short = g["away_team"].split()[-1]
        tip_short = g["tip"].split()[-1]
        winner_short = g["actual_winner"].split()[-1] if g["actual_winner"] != "DRAW" else "DRAW"

        icon = "✅" if g["tip_correct"] else "❌"
        conf = g["confidence"]
        tier = "🔒" if conf >= 0.30 else ("📐" if conf >= 0.10 else "🎲")

        lines.append(
            f"  {icon} {tier} {_esc(home_short)} v {_esc(away_short)}: "
            f"tipped {_esc(tip_short)}, won {_esc(winner_short)} "
            f"({g['actual_margin']:.0f}pts)"
        )

    # Season totals
    if metrics.get("games", 0) > n:
        lines.extend([
            "",
            f"<b>Season:</b> {metrics['tip_correct']}/{metrics['games']} "
            f"({metrics['tip_accuracy']:.1%})",
            f"  vs odds: {metrics['tip_vs_odds']:+d} | "
            f"vs model: {metrics['tip_vs_model']:+d}",
        ])

        # Tier breakdown
        for label, emoji in [("lock", "🔒"), ("lean", "📐"), ("tossup", "🎲")]:
            if f"{label}_games" in metrics:
                lines.append(
                    f"  {emoji} {label.upper()}: "
                    f"{metrics[f'{label}_correct']}/{metrics[f'{label}_games']} "
                    f"({metrics[f'{label}_accuracy']:.0%}), "
                    f"edge {metrics[f'{label}_edge']:+d}"
                )

    send_message("\n".join(lines))


def send_season_report(metrics: dict, year: int, season: pd.DataFrame):
    """Send comprehensive season report."""
    try:
        from scripts.telegram_notify import send_message, _esc
    except ImportError:
        log("  Cannot send Telegram")
        return

    lines = [
        f"📈 <b>{year} Season Dashboard — R{metrics['rounds']}</b>",
        "",
        f"<b>Overall:</b> {metrics['tip_correct']}/{metrics['games']} "
        f"({metrics['tip_accuracy']:.1%})",
        f"  Odds baseline: {metrics['odds_correct']}/{metrics['games']} "
        f"({metrics['odds_accuracy']:.1%})",
        f"  Edge vs odds: <b>{metrics['tip_vs_odds']:+d}</b> tips",
        "",
    ]

    # Per-round scores
    round_scores = []
    for rnd in sorted(season["round"].unique()):
        rm = season[season["round"] == rnd]
        score = rm["tip_correct"].sum()
        total = len(rm)
        round_scores.append(f"{int(score)}")

    lines.append(f"<b>Per round:</b> {' | '.join(round_scores)}")
    lines.append("")

    # Tier breakdown
    for label, emoji in [("lock", "🔒"), ("lean", "📐"), ("tossup", "🎲")]:
        if f"{label}_games" in metrics:
            lines.append(
                f"{emoji} <b>{label.upper()}</b>: "
                f"{metrics[f'{label}_correct']}/{metrics[f'{label}_games']} "
                f"({metrics[f'{label}_accuracy']:.0%}) "
                f"[odds: {metrics.get(f'{label}_odds_accuracy', 0):.0%}] "
                f"edge: {metrics[f'{label}_edge']:+d}"
            )

    if metrics.get("recent_3r_accuracy") is not None:
        lines.extend([
            "",
            f"<b>Recent form (3R):</b> {metrics['recent_3r_accuracy']:.0%} "
            f"(odds: {metrics['recent_3r_odds_accuracy']:.0%})",
        ])

    send_message("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────

def detect_last_completed_round(year: int) -> int | None:
    """Find the most recent round with both predictions and results."""
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    m = matches[(matches["year"] == year) & matches["home_score"].notna()]

    for rnd in range(27, 0, -1):
        pred_path = PREDICTIONS_DIR / f"round_{rnd}_{year}.csv"
        has_results = len(m[m["round"] == str(rnd)]) > 0
        if pred_path.exists() and has_results:
            return rnd
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, help="Round to evaluate")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--season-report", action="store_true")
    parser.add_argument("--no-telegram", action="store_true")
    args = parser.parse_args()

    year = args.year

    if args.season_report:
        log(f"Generating season report for {year}...")
        season = load_season_log(year)
        if season.empty:
            log("  No data yet")
            return
        metrics = compute_metrics(season)
        print(json.dumps(metrics, indent=2, default=str))
        if not args.no_telegram:
            send_season_report(metrics, year, season)
        return

    # Evaluate a specific round
    round_num = args.round
    if round_num is None:
        round_num = detect_last_completed_round(year)
        if round_num is None:
            log(f"No completed rounds found for {year}")
            return

    log(f"Evaluating Round {round_num} ({year})...")

    # Load predictions
    preds = load_predictions(round_num, year)
    if preds is None:
        log(f"  No predictions file for Round {round_num}")
        return

    # Load results
    results = load_results(round_num, year)
    if results.empty:
        log(f"  No results available for Round {round_num}")
        return

    # Match predictions to results
    matched = match_predictions_to_results(preds, results)
    if matched.empty:
        log(f"  No matches could be paired")
        return

    tip_ok = matched["tip_correct"].sum()
    odds_ok = matched["odds_correct"].sum()
    model_ok = matched["model_correct"].sum()
    n = len(matched)

    log(f"  Round {round_num}: {tip_ok}/{n} correct "
        f"(odds: {odds_ok}/{n}, model: {model_ok}/{n})")

    # Save to season log
    save_round_results(year, round_num, matched)

    # Compute season metrics
    season = load_season_log(year)
    metrics = compute_metrics(season)

    log(f"  Season: {metrics['tip_correct']}/{metrics['games']} "
        f"({metrics['tip_accuracy']:.1%}), "
        f"edge vs odds: {metrics['tip_vs_odds']:+d}")

    # Telegram report
    if not args.no_telegram:
        send_round_report(round_num, matched, metrics, year)

    # Check for drift — if we're underperforming odds for 3+ rounds
    if metrics.get("recent_3r_accuracy") is not None:
        recent_edge = metrics["recent_3r_accuracy"] - metrics["recent_3r_odds_accuracy"]
        if recent_edge < -0.05:
            log(f"  ⚠ MODEL DRIFT: recent 3R accuracy ({metrics['recent_3r_accuracy']:.0%}) "
                f"is {recent_edge:.0%} behind odds ({metrics['recent_3r_odds_accuracy']:.0%})")


if __name__ == "__main__":
    main()
