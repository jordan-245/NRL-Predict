"""
NRL Tipping Advisor 2026
========================
Run this every Tuesday after team lists and odds are published.
It categorises each game and tells you exactly what to tip.

Usage:
    python tipping_advisor.py                   # interactive mode
    python tipping_advisor.py --auto            # pull from Odds API + run model
    python tipping_advisor.py --auto --round 5  # specific round

Strategy:
  - LOCK (fav >= 65%): Always tip favourite
  - LEAN (fav 55-65%): Default favourite, flip if model strongly disagrees
  - TOSS-UP (fav < 55%): Use model prediction instead of favourite
  - MARGIN: Always use odds-implied spread, not gut feel
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
#  EDIT THIS SECTION EACH WEEK (manual mode)
# ============================================================
ROUND_NUMBER = 1
SEASON = 2026

# Enter games as: (home_team, away_team, home_odds, away_odds)
# Get odds from Neds/Sportsbet on Tuesday afternoon
ROUND_DATA = [
    # ("Melbourne Storm", "Penrith Panthers", 1.65, 2.30),
    # ("Brisbane Broncos", "Cronulla Sharks", 2.10, 1.80),
    # ... add all 8 games
]

# Optional: if you have model predictions, enter them here
# as home_win_probability (0-1). Leave empty to use odds only.
MODEL_PREDICTIONS = {
    # "Melbourne Storm vs Penrith Panthers": 0.58,
    # "Brisbane Broncos vs Cronulla Sharks": 0.42,
}

# ============================================================
#  THRESHOLDS (don't edit unless tweaking)
# ============================================================

LOCK_THRESHOLD = 0.65      # Above this: always tip favourite
LEAN_THRESHOLD = 0.55      # Above this: default favourite, model can flip
FLIP_CONFIDENCE = 0.52     # Model must give underdog > this to flip a LEAN


# ============================================================
#  CORE LOGIC
# ============================================================

def calculate_implied_prob(home_odds, away_odds):
    """Convert decimal odds to normalised implied probabilities."""
    home_raw = 1 / home_odds
    away_raw = 1 / away_odds
    total = home_raw + away_raw  # includes overround
    return home_raw / total, away_raw / total


def implied_margin(home_odds, away_odds):
    """Estimate the expected margin from the odds spread.

    Calibrated from NRL historical data: prob_diff * 25 gives
    approximate margin in points.
    """
    home_prob, away_prob = calculate_implied_prob(home_odds, away_odds)
    prob_diff = abs(home_prob - away_prob)
    return round(prob_diff * 25, 1)


def categorise_game(fav_prob):
    """Categorise game by closeness."""
    if fav_prob >= LOCK_THRESHOLD:
        return "LOCK"
    elif fav_prob >= LEAN_THRESHOLD:
        return "LEAN"
    else:
        return "TOSS-UP"


def get_tip(home_team, away_team, home_odds, away_odds,
            model_pred=None, spread=None):
    """Determine the tip for a single game.

    Parameters
    ----------
    model_pred : float or None
        Model home_win_probability (0-1).
    spread : float or None
        Bookmaker spread for margin (e.g. -5.5 = home favoured by 5.5).
    """
    home_prob, away_prob = calculate_implied_prob(home_odds, away_odds)
    fav_is_home = home_prob > away_prob
    fav_prob = max(home_prob, away_prob)
    fav_team = home_team if fav_is_home else away_team
    und_team = away_team if fav_is_home else home_team

    category = categorise_game(fav_prob)

    # Margin: prefer bookmaker spread, fallback to odds-implied
    if spread is not None:
        margin = abs(round(spread))
    else:
        margin = implied_margin(home_odds, away_odds)

    # Default: tip the favourite
    tip = fav_team
    reasoning = f"Favourite at {fav_prob:.0%}"
    source = "ODDS"

    if category == "LOCK":
        confidence = 4 if fav_prob >= 0.75 else 3
        reasoning = f"Strong favourite at {fav_prob:.0%} - don't overthink"

    elif category == "LEAN":
        confidence = 2
        reasoning = f"Moderate favourite at {fav_prob:.0%}"

        if model_pred is not None:
            model_home_prob = model_pred
            model_fav_is_home = model_home_prob > 0.5

            if model_fav_is_home != fav_is_home:
                # Model disagrees with odds favourite
                model_und_prob = (model_home_prob if not fav_is_home
                                  else (1 - model_home_prob))
                if model_und_prob > FLIP_CONFIDENCE:
                    tip = und_team
                    source = "MODEL FLIP"
                    confidence = 2
                    reasoning = (
                        f"Model disagrees: gives {und_team} "
                        f"{model_und_prob:.0%} "
                        f"(odds say {fav_team} at {fav_prob:.0%})"
                    )
                else:
                    reasoning += (
                        f" | Model leans {und_team} but not confident "
                        f"enough ({model_und_prob:.0%})"
                    )
            else:
                reasoning += " | Model agrees"
                confidence = 3

    elif category == "TOSS-UP":
        confidence = 1

        if model_pred is not None:
            model_home_prob = model_pred
            if model_home_prob > 0.5:
                tip = home_team
                reasoning = (
                    f"Coin flip ({fav_prob:.0%}) - model picks "
                    f"{home_team} at {model_home_prob:.0%}"
                )
            else:
                tip = away_team
                reasoning = (
                    f"Coin flip ({fav_prob:.0%}) - model picks "
                    f"{away_team} at {1 - model_home_prob:.0%}"
                )
            source = "MODEL"
            confidence = 2
        else:
            reasoning = (
                f"Coin flip ({fav_prob:.0%}) - tipping fav but "
                f"LOW confidence. Run the model!"
            )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "tip": tip,
        "category": category,
        "confidence": confidence,
        "confidence_stars": "*" * confidence + "." * (5 - confidence),
        "reasoning": reasoning,
        "source": source,
        "home_prob": home_prob,
        "away_prob": away_prob,
        "fav_team": fav_team,
        "fav_prob": fav_prob,
        "suggested_margin": margin,
        "home_odds": home_odds,
        "away_odds": away_odds,
    }


# ============================================================
#  OUTPUT
# ============================================================

def print_tips(tips, round_num, year):
    """Print the tipping card."""
    print()
    print("=" * 75)
    print(f"  NRL TIPPING CARD - ROUND {round_num}, {year}")
    print(f"  Generated: {datetime.now().strftime('%A %d %B %Y, %I:%M %p')}")
    print("=" * 75)

    locks = [t for t in tips if t["category"] == "LOCK"]
    leans = [t for t in tips if t["category"] == "LEAN"]
    tossups = [t for t in tips if t["category"] == "TOSS-UP"]

    if locks:
        print(f"\n  LOCKS (tip favourite, don't overthink)")
        print(f"  {'-' * 65}")
        for t in locks:
            print(
                f"  TIP: {t['tip']:<25s} {t['confidence_stars']}  "
                f"({t['home_team']} ${t['home_odds']:.2f} v "
                f"{t['away_team']} ${t['away_odds']:.2f})"
            )
            print(f"       {t['reasoning']}")

    if leans:
        print(f"\n  LEANS (default favourite, check model)")
        print(f"  {'-' * 65}")
        for t in leans:
            flip = " *** FLIPPED ***" if t["source"] == "MODEL FLIP" else ""
            print(
                f"  TIP: {t['tip']:<25s} {t['confidence_stars']}  "
                f"({t['home_team']} ${t['home_odds']:.2f} v "
                f"{t['away_team']} ${t['away_odds']:.2f})"
                f"{flip}"
            )
            print(f"       {t['reasoning']}")

    if tossups:
        print(f"\n  TOSS-UPS (use model, these are coin flips)")
        print(f"  {'-' * 65}")
        for t in tossups:
            print(
                f"  TIP: {t['tip']:<25s} {t['confidence_stars']}  "
                f"({t['home_team']} ${t['home_odds']:.2f} v "
                f"{t['away_team']} ${t['away_odds']:.2f})"
            )
            print(f"       {t['reasoning']}")

    # Summary
    print(f"\n  {'=' * 65}")
    print(f"  FINAL TIPS")
    print(f"  {'=' * 65}")
    print(f"  {'#':<3s} {'Match':<52s} {'Tip':<22s} {'Mar':>4s}")
    print(f"  {'-' * 65}")
    for i, t in enumerate(tips, 1):
        marker = ""
        if t["source"] == "MODEL FLIP":
            marker = " **"
        elif t["source"] == "MODEL":
            marker = " *"
        match_str = f"{t['home_team']} v {t['away_team']}"
        print(
            f"  {i:<3d} {match_str:<52s} "
            f"{t['tip']:<22s} {t['suggested_margin']:>4.0f}{marker}"
        )

    if any(t["source"] in ("MODEL", "MODEL FLIP") for t in tips):
        print(f"\n  * = model pick   ** = model flipped the favourite")

    # Margin suggestion
    margins = [t["suggested_margin"] for t in tips]
    if margins:
        print(f"\n  Avg margin this round: {sum(margins)/len(margins):.0f} pts")
        print(f"  Use individual margins above for tiebreaker (not gut feel)")

    # Game day checklist
    watchlist = leans + tossups
    if watchlist:
        print(f"\n  GAME DAY CHECKLIST:")
        print(f"  [ ] Check final team lists 1 hour before each game")
        print(f"  [ ] If a halfback/fullback/hooker is a late OUT from your "
              f"tipped team:")
        for t in watchlist:
            print(f"      -> {t['home_team']} v {t['away_team']}: "
                  f"flip if key spine player out from {t['tip']}")
        print(f"  [ ] Don't flip any LOCKs unless something extraordinary")

    print("=" * 75)


# ============================================================
#  AUTO MODE: pull from Odds API + run model
# ============================================================

def auto_mode(round_num=None, year=2026):
    """Fetch odds from API, run model, produce tipping card."""
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd

    # Step 1: Load predictions from predict_round.py output
    # First try to use cached predictions, otherwise run the pipeline
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"

    # Run predict_round.py --auto to get fresh predictions
    print("  Running prediction pipeline...")
    print()

    # Import and run the pipeline directly
    import predict_round as pr

    print("=" * 75)
    print(f"  NRL TIPPING ADVISOR {year} (auto mode)")
    print("=" * 75)

    print("\n  STEP 1: Fetching odds from API...")
    try:
        upcoming_api, detected_round = pr.load_upcoming_from_api(round_num, year)
    except (ValueError, Exception) as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    if round_num is None:
        round_num = detected_round

    print(f"  Round {round_num}: {len(upcoming_api)} matches")

    # Step 2: Check for cached models or train
    cp = pr._cache_path(round_num, year)
    cache = pr.load_model_cache(cp)

    if cache is not None:
        print("\n  STEP 2: Scoring with cached models + fresh odds...")
        upcoming_feat = pr._refresh_odds_in_features(
            cache["upcoming_features"], upcoming_api
        )
        results = pr.score_with_models(cache["artifacts"], upcoming_feat)
    else:
        print("\n  STEP 2: Training models (first run this round)...")
        matches, ladders, odds = pr.load_historical_data()
        elo_params = pr.get_elo_params(matches)
        historical, upcoming_feat, feature_cols = pr.build_features(
            matches, ladders, odds, upcoming_api, elo_params
        )
        results, artifacts = pr.train_and_predict(
            historical, upcoming_feat, feature_cols
        )
        pr.save_model_cache(cp, artifacts, upcoming_feat, round_num, year)

    # Step 3: Build tips using model predictions
    tips = []
    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Get decimal odds
        h2h_home = row.get("h2h_home")
        h2h_away = row.get("h2h_away")

        # Fallback: convert from implied prob if no decimal odds
        if pd.isna(h2h_home) or pd.isna(h2h_away):
            odds_hp = row["odds_home_prob"]
            odds_ap = row["odds_away_prob"]
            overround = 1.05  # typical NRL overround
            h2h_home = overround / odds_hp if odds_hp > 0 else 2.0
            h2h_away = overround / odds_ap if odds_ap > 0 else 2.0

        # Model prediction = home_win_prob from OptBlend
        model_pred = row["home_win_prob"]

        # Spread for margin
        spread = row.get("spread_home")
        spread = spread if pd.notna(spread) else None

        tip = get_tip(home, away, h2h_home, h2h_away,
                      model_pred=model_pred, spread=spread)
        tips.append(tip)

    # Save predictions
    pr.save_predictions(results, round_num, year)

    # Print tipping card
    print_tips(tips, round_num, year)


# ============================================================
#  INTERACTIVE MODE
# ============================================================

def interactive_mode():
    """Prompt user to enter games one by one."""
    print(f"\n  Enter games for Round {ROUND_NUMBER}, {SEASON}")
    print(f"  (Type 'done' when finished)\n")

    games = []
    game_num = 1

    while True:
        print(f"  --- Game {game_num} ---")
        home = input("  Home team: ").strip()
        if home.lower() == "done":
            break
        away = input("  Away team: ").strip()

        while True:
            try:
                home_odds = float(input(f"  {home} odds (decimal, e.g. 1.65): "))
                away_odds = float(input(f"  {away} odds (decimal, e.g. 2.30): "))
                break
            except ValueError:
                print("  Invalid odds, try again")

        model_pred = None
        model_input = input(
            f"  Model home_win_prob (Enter to skip): "
        ).strip()
        if model_input:
            try:
                model_pred = float(model_input)
            except ValueError:
                pass

        games.append((home, away, home_odds, away_odds, model_pred))
        game_num += 1
        print()

    return games


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NRL Tipping Advisor 2026"
    )
    parser.add_argument("--auto", action="store_true",
                        help="Fetch odds from API + run model automatically")
    parser.add_argument("--round", type=int, default=None,
                        help="Round number (auto-detected with --auto)")
    parser.add_argument("--year", type=int, default=2026,
                        help="Season year")
    args = parser.parse_args()

    if args.auto:
        auto_mode(round_num=args.round, year=args.year)
        return

    # Manual mode: use ROUND_DATA or interactive
    if ROUND_DATA:
        tips = []
        for game in ROUND_DATA:
            home, away, h_odds, a_odds = game[:4]
            match_key = f"{home} vs {away}"
            model_pred = MODEL_PREDICTIONS.get(match_key)
            tip = get_tip(home, away, h_odds, a_odds, model_pred)
            tips.append(tip)
        print_tips(tips, ROUND_NUMBER, SEASON)
    else:
        games = interactive_mode()
        if not games:
            print("No games entered.")
            return

        tips = []
        for game in games:
            home, away, h_odds, a_odds = game[:4]
            model_pred = game[4] if len(game) > 4 else None
            tip = get_tip(home, away, h_odds, a_odds, model_pred)
            tips.append(tip)
        print_tips(tips, ROUND_NUMBER, SEASON)


if __name__ == "__main__":
    main()
