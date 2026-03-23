#!/usr/bin/env python3
"""
Pre-Kickoff Prediction Refresh
================================
Reruns the full prediction pipeline with fresh data ~90 min before each
game day's first kickoff. Picks up latest weather forecast and odds.

Workflow:
  1. Load week schedule → find today's games
  2. Fetch fresh weather forecast for today's venues (Open-Meteo, free)
  3. Rerun predict_round.py --auto (fast mode: fresh odds + cached model)
  4. Compare new predictions vs current tips
  5. If any tip flips → Telegram alert + auto-resubmit to ESPN Footytips
  6. Log everything

Designed to be called by cron ~90 min before each game day's first kickoff.
Can also be called manually: python scripts/pregame_repredict.py [--dry-run]
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.telegram_notify import send_message, _esc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

AEST = ZoneInfo("Australia/Brisbane")
SCHEDULE_PATH = PROJECT_ROOT / "config" / "week_schedule.json"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
LOG_PATH = PROJECT_ROOT / "logs" / "pregame-repredict.log"


def _log(msg: str):
    """Log to file and stdout."""
    ts = datetime.now(AEST).strftime("%Y-%m-%d %H:%M AEST")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_schedule() -> dict:
    """Load the weekly schedule."""
    if not SCHEDULE_PATH.exists():
        return {}
    with open(SCHEDULE_PATH) as f:
        return json.load(f)


def get_todays_games(schedule: dict) -> list[dict]:
    """Get games scheduled for today (AEST)."""
    today = datetime.now(AEST).strftime("%Y-%m-%d")
    day_info = schedule.get("game_days", {}).get(today, {})
    return day_info.get("games", [])


def get_current_predictions(round_num: int, year: int) -> dict[str, str]:
    """Load current saved predictions. Returns {matchup: tipped_team}."""
    pred_file = PREDICTIONS_DIR / f"round_{round_num}_{year}.csv"
    if not pred_file.exists():
        return {}

    import pandas as pd
    df = pd.read_csv(pred_file)

    tips = {}
    for _, row in df.iterrows():
        matchup = f"{row['home_team']} v {row['away_team']}"
        tip = row.get("tip", row.get("tipped_team", ""))
        prob = row.get("home_win_prob", row.get("prob", 0.5))
        tips[matchup] = {"team": str(tip), "prob": float(prob)}
    return tips


def refresh_weather_forecast(games: list[dict]) -> bool:
    """Fetch fresh weather forecast for today's game venues."""
    try:
        from scraping.open_meteo import fetch_upcoming_weather
        import pandas as pd

        # Build a mini upcoming CSV with today's games
        rows = []
        for g in games:
            # Parse game string: "Team A v Team B (YYYY-MM-DD HH:MM)"
            parts = g.split(" (")
            teams = parts[0] if parts else g
            date_str = parts[1].rstrip(")") if len(parts) > 1 else ""
            home, away = teams.split(" v ") if " v " in teams else (teams, "")

            # Get venue from schedule fixtures
            venue = ""
            schedule = load_schedule()
            for fix in schedule.get("fixtures", []):
                if fix.get("home_team", "") == home.strip():
                    venue = fix.get("venue", "")
                    break

            rows.append({
                "home_team": home.strip(),
                "away_team": away.strip(),
                "venue": venue,
                "date": date_str.split()[0] if date_str else datetime.now(AEST).strftime("%Y-%m-%d"),
            })

        if not rows:
            return False

        # Write temp CSV and fetch forecast
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        df = pd.DataFrame(rows)
        df.to_csv(tmp.name, index=False)
        tmp.close()

        result = fetch_upcoming_weather(tmp.name, delay=0.2)
        os.unlink(tmp.name)

        # Log weather
        for _, row in result.iterrows():
            temp = row.get("temperature_c", "?")
            rain = row.get("precipitation_mm", 0)
            wind = row.get("wind_speed_kmh", 0)
            venue = row.get("venue", "?")
            _log(f"  Weather: {venue} — {temp}°C, {rain}mm rain, {wind}km/h wind")

        return True

    except Exception as e:
        _log(f"  WARNING: Weather refresh failed: {e}")
        return False


def rerun_predictions(dry_run: bool = False) -> bool:
    """Rerun predict_round.py --auto to get fresh predictions."""
    cmd = [sys.executable, str(PROJECT_ROOT / "predict_round.py"), "--auto"]

    _log(f"  Running: {' '.join(cmd)}")
    if dry_run:
        _log("  [DRY RUN] Skipping prediction rerun")
        return True

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300,
        cwd=str(PROJECT_ROOT), env=env,
    )

    if result.returncode != 0:
        _log(f"  ERROR: predict_round.py failed (exit {result.returncode})")
        _log(f"  stderr: {result.stderr[-500:]}")
        return False

    # Extract timing from output
    for line in result.stdout.split("\n"):
        if "Completed in" in line or "Predictions saved" in line:
            _log(f"  {line.strip()}")

    return True


def compare_and_alert(
    old_tips: dict, round_num: int, year: int,
    games: list[dict], dry_run: bool = False,
) -> list[dict]:
    """Compare old tips with new predictions. Return list of flips."""
    new_tips = get_current_predictions(round_num, year)
    if not new_tips:
        _log("  WARNING: No new predictions found")
        return []

    flips = []
    for matchup, new in new_tips.items():
        old = old_tips.get(matchup)
        if old is None:
            continue

        # Check if this is one of today's games
        is_today = any(matchup.replace(" v ", " v ") in g for g in games)
        if not is_today:
            continue

        old_team = old["team"]
        new_team = new["team"]
        new_prob = new["prob"]
        old_prob = old["prob"]

        if old_team != new_team:
            flips.append({
                "matchup": matchup,
                "old_tip": old_team,
                "new_tip": new_team,
                "old_prob": old_prob,
                "new_prob": new_prob,
            })
            _log(f"  🔄 FLIP: {matchup}")
            _log(f"     {old_team} ({old_prob:.0%}) → {new_team} ({new_prob:.0%})")
        else:
            prob_delta = abs(new_prob - old_prob)
            if prob_delta > 0.05:
                _log(f"  📊 SHIFT: {matchup} — {old_prob:.0%} → {new_prob:.0%} (same tip: {new_team})")

    return flips


def send_flip_alert(flips: list[dict], games: list[dict], round_num: int):
    """Send Telegram alert for flipped tips."""
    if not flips:
        return

    lines = [
        f"🔄 <b>Pre-Kickoff Tip Update — Round {round_num}</b>",
        f"📅 {datetime.now(AEST).strftime('%a %d %b, %I:%M %p AEST')}",
        "",
    ]

    for flip in flips:
        lines.append(f"<b>{_esc(flip['matchup'])}</b>")
        lines.append(
            f"  ❌ {_esc(flip['old_tip'])} ({flip['old_prob']:.0%})"
            f" → ✅ {_esc(flip['new_tip'])} ({flip['new_prob']:.0%})"
        )
        lines.append("")

    lines.append(f"💡 Fresh odds + weather forecast picked up new signal")
    lines.append(f"🎯 {len(flips)} tip(s) updated")

    send_message("\n".join(lines))


def auto_resubmit(flips: list[dict], round_num: int, dry_run: bool = False):
    """Resubmit flipped tips to ESPN Footytips."""
    if not flips or dry_run:
        return

    try:
        from scripts.footytips_submit import get_auth_headers, submit_tips, load_predictions
        from scripts.pregame_check import resubmit_tip

        headers = get_auth_headers()
        if not headers:
            _log("  WARNING: No ESPN auth — cannot resubmit")
            return

        import pandas as pd
        year = datetime.now(AEST).year
        pred_file = PREDICTIONS_DIR / f"round_{round_num}_{year}.csv"
        if not pred_file.exists():
            return

        preds = pd.read_csv(pred_file)

        for flip in flips:
            new_team = flip["new_tip"]
            # Find the event in predictions
            match_row = preds[
                preds.apply(lambda r: f"{r['home_team']} v {r['away_team']}" == flip["matchup"], axis=1)
            ]
            if match_row.empty:
                continue

            row = match_row.iloc[0]
            event = {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
            }
            ok = resubmit_tip(event, new_team, round_num, headers)
            _log(f"  {'✅' if ok else '❌'} Resubmitted: {flip['matchup']} → {new_team}")

    except Exception as e:
        _log(f"  ERROR resubmitting: {e}")


def send_no_change_summary(games: list[dict], round_num: int):
    """Send brief Telegram confirmation when no tips changed."""
    n = len(games)
    now = datetime.now(AEST).strftime("%I:%M %p")
    lines = [
        f"✅ <b>Pre-Kickoff Check — Round {round_num}</b>",
        f"⏰ {now} AEST | {n} game{'s' if n != 1 else ''} today",
        f"📊 Fresh odds + weather checked — all tips confirmed",
    ]
    send_message("\n".join(lines), silent=True)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Pre-kickoff prediction refresh")
    p.add_argument("--dry-run", action="store_true", help="Check schedule only, don't rerun predictions")
    p.add_argument("--force", action="store_true", help="Run even if no games today")
    p.add_argument("--no-submit", action="store_true", help="Don't auto-resubmit flipped tips")
    p.add_argument("--quiet", action="store_true", help="No Telegram on no-change")
    args = p.parse_args()

    _log("=" * 60)
    _log("Pre-Kickoff Prediction Refresh")
    _log("=" * 60)

    # 1. Load schedule and find today's games
    schedule = load_schedule()
    if not schedule:
        _log("ERROR: No week schedule found. Run plan_week.py first.")
        return 1

    round_num = schedule.get("round", 0)
    year = datetime.now(AEST).year
    games = get_todays_games(schedule)

    if not games and not args.force:
        _log(f"No games today. Round {round_num}.")
        return 0

    _log(f"Round {round_num} — {len(games)} game(s) today:")
    for g in games:
        _log(f"  🏉 {g}")

    # 2. Save current predictions (for comparison)
    old_tips = get_current_predictions(round_num, year)
    _log(f"Current predictions loaded: {len(old_tips)} tips")

    # 3. Refresh weather forecast
    _log("\nStep 1: Refreshing weather forecast...")
    refresh_weather_forecast(games)

    # 4. Rerun predictions with fresh data
    _log("\nStep 2: Rerunning predictions with fresh odds + weather...")
    ok = rerun_predictions(dry_run=args.dry_run)
    if not ok:
        _log("FAILED: Prediction rerun failed")
        send_message("❌ <b>Pre-Kickoff Refresh FAILED</b>\n\npredict_round.py errored. Check logs.")
        return 1

    # 5. Compare and detect flips
    _log("\nStep 3: Comparing predictions...")
    flips = compare_and_alert(old_tips, round_num, year, games, dry_run=args.dry_run)

    # 6. Alert and resubmit
    if flips:
        _log(f"\n🔄 {len(flips)} tip(s) flipped!")
        send_flip_alert(flips, games, round_num)
        if not args.no_submit and not args.dry_run:
            _log("\nStep 4: Auto-resubmitting to ESPN...")
            auto_resubmit(flips, round_num, dry_run=args.dry_run)
    else:
        _log("\n✅ All tips confirmed — no changes")
        if not args.quiet and not args.dry_run:
            send_no_change_summary(games, round_num)

    _log(f"\nDone. ({len(games)} games, {len(flips)} flips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
