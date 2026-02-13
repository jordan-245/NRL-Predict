"""
NRL Tipping Tracker 2026
========================
Track weekly results and view season dashboard.

Usage:
    python tipping_tracker.py --record 1          # record results for round 1
    python tipping_tracker.py --record 1 --auto   # auto-fetch scores from API
    python tipping_tracker.py                     # show season dashboard

Weekly workflow:
    1. Tuesday:  python tipping_advisor.py --auto  (generate tips)
    2. Tuesday:  python tipping_tracker.py --record N --auto  (log last round)
    3. Anytime:  python tipping_tracker.py  (check season progress)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = LOG_DIR / "predictions"

# 2025 benchmark: Jordan got 136/213 (63.8%), favourite got 133/213 (62.4%)
# ~8 games per round, ~26.6 rounds
BENCHMARK_2025_YOU = 136
BENCHMARK_2025_FAV = 133
BENCHMARK_2025_TOTAL = 213
BENCHMARK_2025_ROUNDS = 27  # approximate


def log_path(year: int) -> Path:
    return LOG_DIR / f"tipping_log_{year}.csv"


def load_log(year: int) -> pd.DataFrame:
    """Load the tipping log CSV, or return empty DataFrame."""
    path = log_path(year)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=[
        "season", "round", "home_team", "away_team",
        "category", "tip", "model_pick", "odds_fav",
        "home_score", "away_score", "winner",
        "tip_correct", "model_correct", "fav_correct",
        "model_agreed_odds", "h2h_home", "h2h_away",
        "model_home_prob", "odds_home_prob",
    ])


def save_log(df: pd.DataFrame, year: int):
    path = log_path(year)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Log saved to {path}")


def load_predictions(round_num: int, year: int) -> pd.DataFrame | None:
    """Load saved predictions for a round."""
    path = PREDICTIONS_DIR / f"round_{round_num}_{year}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def categorise_from_predictions(preds: pd.DataFrame) -> list[dict]:
    """Re-derive LOCK/LEAN/TOSS-UP categories from saved predictions."""
    from tipping_advisor import get_tip

    games = []
    for _, row in preds.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        h2h_home = row.get("h2h_home")
        h2h_away = row.get("h2h_away")

        if pd.isna(h2h_home) or pd.isna(h2h_away):
            odds_hp = row["odds_home_prob"]
            odds_ap = row["odds_away_prob"]
            h2h_home = 1.05 / odds_hp if odds_hp > 0 else 2.0
            h2h_away = 1.05 / odds_ap if odds_ap > 0 else 2.0

        model_pred = row["home_win_prob"]
        spread = row.get("spread_home")
        spread = spread if pd.notna(spread) else None

        tip_info = get_tip(home, away, h2h_home, h2h_away,
                           model_pred=model_pred, spread=spread)

        odds_hp = row["odds_home_prob"]
        odds_fav = home if odds_hp > 0.5 else away
        model_pick = home if model_pred > 0.5 else away

        games.append({
            "home_team": home,
            "away_team": away,
            "category": tip_info["category"],
            "tip": tip_info["tip"],
            "model_pick": model_pick,
            "odds_fav": odds_fav,
            "model_agreed_odds": 1 if model_pick == odds_fav else 0,
            "h2h_home": h2h_home,
            "h2h_away": h2h_away,
            "model_home_prob": model_pred,
            "odds_home_prob": odds_hp,
        })
    return games


def fetch_scores_auto(round_num: int, year: int) -> dict[str, tuple[int, int]]:
    """Try to fetch match scores automatically.

    Returns dict of {(home, away): (home_score, away_score)}.
    Tries matches.parquet first (free), then Odds API scores (2 credits).
    """
    scores = {}

    # Try matches.parquet first
    matches_path = PROJECT_ROOT / "data" / "processed" / "matches.parquet"
    if matches_path.exists():
        matches = pd.read_parquet(matches_path)
        round_matches = matches[
            (matches["year"] == year) &
            (matches["round"].astype(str) == str(round_num))
        ]
        for _, row in round_matches.iterrows():
            hs = row.get("home_score")
            aws = row.get("away_score")
            if pd.notna(hs) and pd.notna(aws):
                key = (row["home_team"], row["away_team"])
                scores[key] = (int(hs), int(aws))

    if scores:
        print(f"  Found {len(scores)} scores from matches.parquet")
        return scores

    # Try Odds API scores
    try:
        from scraping.odds_api import get_scores
        from config.team_mappings import standardise_team_name
        print("  Fetching scores from Odds API (2 credits)...")
        raw_scores = get_scores(days_from=7)
        for event in raw_scores:
            if not event.get("completed"):
                continue
            home_raw = event["home_team"]
            away_raw = event["away_team"]
            try:
                home = standardise_team_name(home_raw)
                away = standardise_team_name(away_raw)
            except KeyError:
                home, away = home_raw, away_raw
            for s in event.get("scores", []):
                if s["name"] == home_raw:
                    hs = int(s["score"])
                elif s["name"] == away_raw:
                    aws = int(s["score"])
            scores[(home, away)] = (hs, aws)
        if scores:
            print(f"  Found {len(scores)} completed scores from API")
    except Exception as e:
        print(f"  Could not fetch from API: {e}")

    return scores


def record_round(round_num: int, year: int, auto: bool):
    """Record results for a completed round."""
    print()
    print("=" * 70)
    print(f"  RECORDING RESULTS - ROUND {round_num}, {year}")
    print("=" * 70)

    # Load predictions
    preds = load_predictions(round_num, year)
    if preds is None:
        print(f"\n  No predictions found for Round {round_num}.")
        print(f"  Expected: {PREDICTIONS_DIR / f'round_{round_num}_{year}.csv'}")
        print(f"  Run tipping_advisor.py --auto first.")
        sys.exit(1)

    # Get categories and tips
    games = categorise_from_predictions(preds)
    print(f"\n  Found {len(games)} predictions for Round {round_num}")

    # Get scores
    scores = {}
    if auto:
        scores = fetch_scores_auto(round_num, year)

    # Match scores to games, prompt for missing
    log = load_log(year)

    # Check if round already logged
    existing = log[(log["season"] == year) & (log["round"] == round_num)]
    if len(existing) > 0:
        print(f"\n  Round {round_num} already has {len(existing)} entries in log.")
        resp = input("  Overwrite? (y/n): ").strip().lower()
        if resp != "y":
            print("  Skipped.")
            return
        log = log[~((log["season"] == year) & (log["round"] == round_num))]

    new_rows = []
    print()
    for i, g in enumerate(games):
        home = g["home_team"]
        away = g["away_team"]

        # Find score
        hs, aws = None, None
        for (sh, sa), (score_h, score_a) in scores.items():
            if sh == home and sa == away:
                hs, aws = score_h, score_a
                break
            elif sh == away and sa == home:
                hs, aws = score_a, score_h
                break

        if hs is None or aws is None:
            print(f"  {home} vs {away}  [{g['category']}]")
            print(f"    Tip: {g['tip']}")
            while True:
                try:
                    score_input = input(f"    Score ({home} - {away}): ").strip()
                    if "-" in score_input:
                        parts = score_input.split("-")
                        hs, aws = int(parts[0].strip()), int(parts[1].strip())
                    else:
                        hs = int(input(f"    {home} score: "))
                        aws = int(input(f"    {away} score: "))
                    break
                except (ValueError, IndexError):
                    print("    Invalid. Enter as 'HS - AS' (e.g. '24 - 18')")
        else:
            print(f"  {home} {hs} - {aws} {away}  [{g['category']}]")

        # Determine winner
        if hs > aws:
            winner = home
        elif aws > hs:
            winner = away
        else:
            winner = "DRAW"

        tip_correct = 1 if g["tip"] == winner else 0
        model_correct = 1 if g["model_pick"] == winner else 0
        fav_correct = 1 if g["odds_fav"] == winner else 0

        marker = "OK" if tip_correct else "WRONG"
        if g["tip"] != g["odds_fav"]:
            if tip_correct:
                marker = "OK (upset pick!)"
            else:
                marker = "WRONG (upset pick failed)"

        print(f"    Winner: {winner}  |  Tip: {g['tip']}  →  {marker}")

        new_rows.append({
            "season": year,
            "round": round_num,
            "home_team": home,
            "away_team": away,
            "category": g["category"],
            "tip": g["tip"],
            "model_pick": g["model_pick"],
            "odds_fav": g["odds_fav"],
            "home_score": hs,
            "away_score": aws,
            "winner": winner,
            "tip_correct": tip_correct,
            "model_correct": model_correct,
            "fav_correct": fav_correct,
            "model_agreed_odds": g["model_agreed_odds"],
            "h2h_home": g["h2h_home"],
            "h2h_away": g["h2h_away"],
            "model_home_prob": g["model_home_prob"],
            "odds_home_prob": g["odds_home_prob"],
        })

    new_df = pd.DataFrame(new_rows)
    log = pd.concat([log, new_df], ignore_index=True)
    log = log.sort_values(["season", "round"]).reset_index(drop=True)
    save_log(log, year)

    # Print round summary
    n = len(new_rows)
    tips_right = sum(r["tip_correct"] for r in new_rows)
    fav_right = sum(r["fav_correct"] for r in new_rows)
    model_right = sum(r["model_correct"] for r in new_rows)

    print(f"\n  ROUND {round_num} SUMMARY")
    print(f"  {'-' * 40}")
    print(f"  Your tips:  {tips_right}/{n}")
    print(f"  Favourite:  {fav_right}/{n}")
    print(f"  Model:      {model_right}/{n}")

    # Show disagreements
    disagrees = [r for r in new_rows if r["model_pick"] != r["odds_fav"]]
    if disagrees:
        print(f"\n  Model disagreed with odds on {len(disagrees)} game(s):")
        for r in disagrees:
            correct_marker = "RIGHT" if r["model_correct"] else "WRONG"
            print(f"    {r['home_team']} v {r['away_team']}: "
                  f"model={r['model_pick']}, fav={r['odds_fav']} → {correct_marker}")


def show_dashboard(year: int):
    """Show season-long dashboard."""
    log = load_log(year)

    if log.empty:
        print(f"\n  No results logged for {year} yet.")
        print(f"  After each round: python tipping_tracker.py --record N")
        return

    # Exclude draws from accuracy calculations
    log = log[log["winner"] != "DRAW"].copy()

    total = len(log)
    rounds_played = log["round"].nunique()
    tips_right = int(log["tip_correct"].sum())
    fav_right = int(log["fav_correct"].sum())
    model_right = int(log["model_correct"].sum())

    print()
    print("=" * 70)
    print(f"  NRL {year} TIPPING DASHBOARD")
    print(f"  Through Round {int(log['round'].max())} "
          f"({rounds_played} rounds, {total} games)")
    print("=" * 70)

    # Overall
    print(f"\n  OVERALL ACCURACY")
    print(f"  {'-' * 50}")
    print(f"  {'Your tips:':<25s} {tips_right:>3d}/{total}  "
          f"({tips_right/total*100:.1f}%)")
    print(f"  {'Favourite baseline:':<25s} {fav_right:>3d}/{total}  "
          f"({fav_right/total*100:.1f}%)")
    print(f"  {'Model picks:':<25s} {model_right:>3d}/{total}  "
          f"({model_right/total*100:.1f}%)")
    print(f"  {'Edge vs favourite:':<25s} {tips_right - fav_right:>+3d} tips")

    # By category
    print(f"\n  BY CATEGORY")
    print(f"  {'-' * 50}")
    for cat in ["LOCK", "LEAN", "TOSS-UP"]:
        cat_games = log[log["category"] == cat]
        if cat_games.empty:
            continue
        n = len(cat_games)
        tip_acc = int(cat_games["tip_correct"].sum())
        fav_acc = int(cat_games["fav_correct"].sum())
        model_acc = int(cat_games["model_correct"].sum())
        print(f"  {cat:<10s} ({n:>2d} games):  "
              f"Tips {tip_acc}/{n} ({tip_acc/n*100:.0f}%)  "
              f"Fav {fav_acc}/{n} ({fav_acc/n*100:.0f}%)  "
              f"Model {model_acc}/{n} ({model_acc/n*100:.0f}%)")

    # Model disagreement analysis (the valuable signal)
    disagree = log[log["model_agreed_odds"] == 0]
    if not disagree.empty:
        n_dis = len(disagree)
        model_won = int(disagree["model_correct"].sum())
        fav_won = int(disagree["fav_correct"].sum())
        print(f"\n  MODEL DISAGREED WITH ODDS ({n_dis} games)")
        print(f"  {'-' * 50}")
        print(f"  Model was right:     {model_won}/{n_dis} "
              f"({model_won/n_dis*100:.0f}%)")
        print(f"  Favourite was right: {fav_won}/{n_dis} "
              f"({fav_won/n_dis*100:.0f}%)")
        if model_won > fav_won:
            print(f"  --> Model adding value on disagreements (+{model_won - fav_won})")
        elif model_won < fav_won:
            print(f"  --> Model LOSING value on disagreements ({model_won - fav_won})")
        else:
            print(f"  --> Model neutral on disagreements (same as favourite)")
    else:
        print(f"\n  No model disagreements with odds yet.")

    # 2025 pace comparison
    print(f"\n  2025 PACE COMPARISON")
    print(f"  {'-' * 50}")
    pace_2025_you = BENCHMARK_2025_YOU / BENCHMARK_2025_TOTAL
    pace_2025_fav = BENCHMARK_2025_FAV / BENCHMARK_2025_TOTAL
    projected_you = round(tips_right / total * BENCHMARK_2025_TOTAL)
    projected_fav = round(fav_right / total * BENCHMARK_2025_TOTAL)
    games_per_round = total / rounds_played if rounds_played > 0 else 8
    expected_at_this_point_you = round(pace_2025_you * total)
    expected_at_this_point_fav = round(pace_2025_fav * total)

    print(f"  {'':20s} {'2026 (actual)':>15s} {'2025 pace':>12s}")
    print(f"  {'Your tips:':<20s} {tips_right:>7d}/{total:<6d} "
          f"{expected_at_this_point_you:>5d}/{total}")
    print(f"  {'Favourite:':<20s} {fav_right:>7d}/{total:<6d} "
          f"{expected_at_this_point_fav:>5d}/{total}")
    if tips_right > expected_at_this_point_you:
        print(f"  --> Ahead of 2025 pace by "
              f"{tips_right - expected_at_this_point_you} tips")
    elif tips_right < expected_at_this_point_you:
        print(f"  --> Behind 2025 pace by "
              f"{expected_at_this_point_you - tips_right} tips")
    else:
        print(f"  --> On pace with 2025")

    print(f"\n  Projected season total: ~{projected_you} tips correct "
          f"(2025: {BENCHMARK_2025_YOU})")

    # Round-by-round
    print(f"\n  ROUND-BY-ROUND")
    print(f"  {'-' * 50}")
    print(f"  {'Rd':<4s} {'Tips':>5s} {'Fav':>5s} {'Model':>6s} "
          f"{'Cumul':>6s} {'Cum%':>5s}")
    cumul = 0
    cumul_total = 0
    for rnd in sorted(log["round"].unique()):
        rnd_games = log[log["round"] == rnd]
        n = len(rnd_games)
        t = int(rnd_games["tip_correct"].sum())
        f = int(rnd_games["fav_correct"].sum())
        m = int(rnd_games["model_correct"].sum())
        cumul += t
        cumul_total += n
        pct = cumul / cumul_total * 100
        print(f"  {int(rnd):<4d} {t:>2d}/{n:<2d} {f:>2d}/{n:<2d} "
              f"{m:>3d}/{n:<2d}  {cumul:>3d}/{cumul_total:<3d} {pct:>5.1f}%")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="NRL Tipping Tracker 2026"
    )
    parser.add_argument("--record", type=int, default=None,
                        help="Record results for round N")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-fetch scores (from parquet or API)")
    parser.add_argument("--year", type=int, default=2026,
                        help="Season year")
    args = parser.parse_args()

    if args.record is not None:
        record_round(args.record, args.year, auto=args.auto)
        # Show dashboard after recording
        show_dashboard(args.year)
    else:
        show_dashboard(args.year)


if __name__ == "__main__":
    main()
