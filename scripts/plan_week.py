#!/usr/bin/env python3
"""
NRL Weekly Schedule Planner
============================
Fetches the upcoming round's kickoff times and writes a schedule file
that the cron system uses to set pregame check windows.

Called by Monday refresh. Also updates crontab pregame entries to match
the actual game days for the week.

Usage:
    python scripts/plan_week.py              # auto-detect next round
    python scripts/plan_week.py --round 5    # specific round
    python scripts/plan_week.py --dry-run    # show plan without writing cron

Output:
    config/week_schedule.json — game times, pregame windows, tip deadline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Auto-load .env
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

AEST = timezone(timedelta(hours=10))
SCHEDULE_PATH = PROJECT_ROOT / "config" / "week_schedule.json"

# Pregame checks start this many hours before first kickoff of the day
PREGAME_LEAD_HOURS = 2
# Pregame checks end this many minutes after last kickoff of the day
PREGAME_TRAIL_MINUTES = 30
# Minimum hours before first kickoff that tips must be submitted
TIP_DEADLINE_HOURS = 3


def fetch_kickoff_times(round_num: int | None = None) -> tuple[list[dict], int]:
    """Fetch upcoming round fixtures with kickoff times from Odds API.

    Returns (fixtures_list, round_num).  Each fixture has:
        home_team, away_team, kickoff_utc, kickoff_aest, day_of_week
    """
    from scraping.odds_api import get_events, detect_next_round

    events = get_events()
    if not events:
        raise ValueError("No upcoming NRL events. Season may not have started.")

    if round_num is None:
        round_num = detect_next_round(events)

    # Filter to this round's window (same logic as get_upcoming_round)
    import pandas as pd
    times = []
    for e in events:
        ct = e.get("commence_time")
        if ct:
            times.append((pd.to_datetime(ct, utc=True), e))
    times.sort(key=lambda x: x[0])

    if times:
        round_start = times[0][0]
        round_end = round_start + pd.Timedelta(days=10)
        round_events = [(t, e) for t, e in times if t <= round_end][:8]
    else:
        round_events = [(pd.NaT, e) for e in events[:8]]

    fixtures = []
    for ts, event in round_events:
        ko_utc = ts.to_pydatetime() if not pd.isna(ts) else None
        ko_aest = ko_utc.astimezone(AEST) if ko_utc else None

        from scraping.odds_api import _standardise_api_team
        fixtures.append({
            "home_team": _standardise_api_team(event["home_team"]),
            "away_team": _standardise_api_team(event["away_team"]),
            "kickoff_utc": ko_utc.isoformat() if ko_utc else None,
            "kickoff_aest": ko_aest.strftime("%Y-%m-%d %H:%M") if ko_aest else None,
            "day_of_week": ko_aest.strftime("%A") if ko_aest else None,
            "day_num": ko_aest.weekday() if ko_aest else None,  # 0=Mon
        })

    return fixtures, round_num


def build_schedule(fixtures: list[dict], round_num: int) -> dict:
    """Build the weekly schedule from fixture kickoff times.

    Returns a dict with:
        round, first_kickoff, tip_deadline, game_days, pregame_windows,
        cron_pregame_spec, fixtures
    """
    if not fixtures or not fixtures[0].get("kickoff_aest"):
        return {"round": round_num, "error": "No kickoff times available"}

    # Parse kickoff times
    kickoffs = []
    for f in fixtures:
        if f["kickoff_aest"]:
            ko = datetime.strptime(f["kickoff_aest"], "%Y-%m-%d %H:%M")
            ko = ko.replace(tzinfo=AEST)
            kickoffs.append(ko)

    kickoffs.sort()
    first_kickoff = kickoffs[0]
    last_kickoff = kickoffs[-1]

    # Tip deadline: tips must be in BEFORE first kickoff
    tip_deadline = first_kickoff - timedelta(hours=TIP_DEADLINE_HOURS)

    # Group games by day
    game_days = {}  # day_name -> {date, first_ko, last_ko, games}
    for f in fixtures:
        if not f["kickoff_aest"]:
            continue
        ko = datetime.strptime(f["kickoff_aest"], "%Y-%m-%d %H:%M")
        day_key = ko.strftime("%Y-%m-%d")
        day_name = f["day_of_week"]
        if day_key not in game_days:
            game_days[day_key] = {
                "date": day_key,
                "day_name": day_name,
                "day_num": f["day_num"],
                "first_ko": f["kickoff_aest"],
                "last_ko": f["kickoff_aest"],
                "n_games": 0,
                "games": [],
            }
        game_days[day_key]["last_ko"] = f["kickoff_aest"]
        game_days[day_key]["n_games"] += 1
        game_days[day_key]["games"].append(
            f"{f['home_team']} v {f['away_team']} ({f['kickoff_aest']})"
        )

    # Build pregame check windows per day
    # Checks should run from PREGAME_LEAD_HOURS before first KO
    # until PREGAME_TRAIL_MINUTES after last KO
    pregame_windows = {}
    for day_key, info in sorted(game_days.items()):
        first_ko = datetime.strptime(info["first_ko"], "%Y-%m-%d %H:%M")
        last_ko = datetime.strptime(info["last_ko"], "%Y-%m-%d %H:%M")
        check_start = first_ko - timedelta(hours=PREGAME_LEAD_HOURS)
        check_end = last_ko + timedelta(minutes=PREGAME_TRAIL_MINUTES)
        pregame_windows[day_key] = {
            "day_name": info["day_name"],
            "check_start": check_start.strftime("%H:%M"),
            "check_end": check_end.strftime("%H:%M"),
            "start_hour": check_start.hour,
            "end_hour": min(check_end.hour + 1, 23),  # ceil to next hour
        }

    # Build cron spec for pregame checks
    # Collect unique (day_num, hour_range) pairs
    cron_days = set()
    hour_min = 23
    hour_max = 0
    for day_key, info in game_days.items():
        # cron uses 0=Sun, 1=Mon, ... 6=Sat
        # Python weekday() uses 0=Mon, ... 6=Sun
        py_day = info["day_num"]
        cron_day = (py_day + 1) % 7  # convert: Mon=1, Tue=2, ... Sun=0
        cron_days.add(cron_day)

        window = pregame_windows[day_key]
        hour_min = min(hour_min, window["start_hour"])
        hour_max = max(hour_max, window["end_hour"])

    cron_days_str = ",".join(str(d) for d in sorted(cron_days))
    cron_spec = f"*/30 {hour_min}-{hour_max} * * {cron_days_str}"

    # Check if the NEXT Tuesday at 5pm is early enough for tips.
    # This script runs on Monday. The tip cron fires next day (Tuesday 5pm).
    # But some rounds start before Tuesday (e.g. Las Vegas Sunday games).
    now = datetime.now(AEST)
    # Find the next Tuesday from now
    days_until_tuesday = (1 - now.weekday()) % 7  # 1=Tuesday
    if days_until_tuesday == 0 and now.hour >= 17:
        days_until_tuesday = 7  # already past this Tuesday 5pm
    next_tuesday_5pm = (now + timedelta(days=days_until_tuesday)).replace(
        hour=17, minute=0, second=0, microsecond=0
    )

    if next_tuesday_5pm >= first_kickoff:
        # Tuesday 5pm is AFTER first kickoff — need earlier submission!
        tip_warning = (
            f"WARNING: First kickoff {first_kickoff.strftime('%A %d %b %H:%M')} is "
            f"BEFORE Tuesday 5pm ({next_tuesday_5pm.strftime('%d %b')}). "
            f"Tips must be submitted before {tip_deadline.strftime('%A %d %b %H:%M')}!"
        )
    else:
        hours_before = (first_kickoff - next_tuesday_5pm).total_seconds() / 3600
        tip_warning = None if hours_before >= TIP_DEADLINE_HOURS else (
            f"CAUTION: Only {hours_before:.1f}h between Tuesday 5pm "
            f"and first kickoff {first_kickoff.strftime('%A %d %b %H:%M')}"
        )

    # Build per-game-day repredict schedule
    # Fires ~90 min before each day's first kickoff
    REPREDICT_LEAD_MINUTES = 90
    repredict_times = {}
    for day_key, info in sorted(game_days.items()):
        first_ko = datetime.strptime(info["first_ko"], "%Y-%m-%d %H:%M")
        repredict_at = first_ko - timedelta(minutes=REPREDICT_LEAD_MINUTES)
        repredict_times[day_key] = {
            "time": repredict_at.strftime("%H:%M"),
            "hour": repredict_at.hour,
            "minute": repredict_at.minute,
            "first_kickoff": info["first_ko"],
            "day_name": info["day_name"],
        }

    schedule = {
        "round": round_num,
        "generated_at": datetime.now(AEST).strftime("%Y-%m-%d %H:%M AEST"),
        "first_kickoff": first_kickoff.strftime("%Y-%m-%d %H:%M AEST"),
        "last_kickoff": last_kickoff.strftime("%Y-%m-%d %H:%M AEST"),
        "tip_deadline": tip_deadline.strftime("%Y-%m-%d %H:%M AEST"),
        "tip_warning": tip_warning,
        "game_days": game_days,
        "pregame_windows": pregame_windows,
        "repredict_times": repredict_times,
        "cron_pregame_spec": cron_spec,
        "cron_pregame_days": cron_days_str,
        "cron_pregame_hours": f"{hour_min}-{hour_max}",
        "fixtures": fixtures,
    }

    return schedule


def update_crontab(cron_spec: str, dry_run: bool = False) -> bool:
    """Update the pregame cron entry to match this week's schedule.

    Only modifies the NRL pregame line; all other cron entries are preserved.
    """
    try:
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, check=True
        )
        current = result.stdout
    except subprocess.CalledProcessError:
        print("  ⚠ No existing crontab found")
        return False

    # Find and replace the NRL pregame line
    # Pattern: */30 <hours> * * <days> /root/NRL-Predict/scripts/nrl-cron.sh pregame
    pattern = r"^[^\n#]*nrl-cron\.sh pregame.*$"
    new_line = f"{cron_spec} /root/NRL-Predict/scripts/nrl-cron.sh pregame"

    # Also update the comment above it
    comment_pattern = r"^# (?:Thu-Sun|Pregame).*(?:pregame|lineup|pre-kickoff|auto-set).*$"
    lines = current.split("\n")
    new_lines = []
    i = 0
    replaced = False
    while i < len(lines):
        line = lines[i]
        if re.match(comment_pattern, line):
            # Replace comment with updated schedule info
            new_lines.append(f"# Pregame checks: cron={cron_spec} (auto-set by plan_week.py)")
            i += 1
            continue
        if re.match(pattern, line):
            new_lines.append(new_line)
            replaced = True
            i += 1
            continue
        new_lines.append(line)
        i += 1

    if not replaced:
        print("  ⚠ Could not find pregame cron line to update")
        return False

    new_crontab = "\n".join(new_lines)

    if dry_run:
        print(f"\n  Would update pregame cron to: {cron_spec}")
        print(f"  Full line: {new_line}")
        return True

    proc = subprocess.run(
        ["crontab", "-"], input=new_crontab, capture_output=True, text=True
    )
    if proc.returncode != 0:
        print(f"  ✗ Failed to update crontab: {proc.stderr}")
        return False

    print(f"  ✓ Pregame cron updated: {cron_spec}")
    return True


def print_schedule(schedule: dict):
    """Pretty-print the weekly schedule."""
    print()
    print("=" * 70)
    print(f"  NRL WEEKLY SCHEDULE — ROUND {schedule['round']}")
    print(f"  Generated: {schedule['generated_at']}")
    print("=" * 70)

    if "error" in schedule:
        print(f"\n  ⚠ {schedule['error']}")
        return

    print(f"\n  First kickoff:  {schedule['first_kickoff']}")
    print(f"  Last kickoff:   {schedule['last_kickoff']}")
    print(f"  Tip deadline:   {schedule['tip_deadline']}")

    if schedule.get("tip_warning"):
        print(f"\n  ⚠ {schedule['tip_warning']}")

    print(f"\n  GAME DAYS:")
    print(f"  {'-' * 60}")
    for day_key, info in sorted(schedule["game_days"].items()):
        window = schedule["pregame_windows"].get(day_key, {})
        print(f"  {info['day_name']:10s} {day_key}  "
              f"({info['n_games']} game{'s' if info['n_games'] > 1 else ''})  "
              f"checks {window.get('check_start', '?')}-{window.get('check_end', '?')}")
        for game in info["games"]:
            print(f"    • {game}")

    print(f"\n  PREGAME CRON: {schedule['cron_pregame_spec']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="NRL Weekly Schedule Planner")
    parser.add_argument("--round", type=int, default=None, help="Round number")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without updating cron")
    args = parser.parse_args()

    print("\n  Fetching upcoming round kickoff times...")
    try:
        fixtures, round_num = fetch_kickoff_times(args.round)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        sys.exit(1)

    print(f"  Round {round_num}: {len(fixtures)} fixtures")

    schedule = build_schedule(fixtures, round_num)
    print_schedule(schedule)

    # Save schedule
    SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEDULE_PATH, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"\n  Schedule saved to {SCHEDULE_PATH}")

    # Update crontab — pregame checks + per-day repredict triggers
    if "error" not in schedule and schedule.get("cron_pregame_spec"):
        update_crontab(schedule["cron_pregame_spec"], dry_run=args.dry_run)
        install_repredict_cron(schedule, dry_run=args.dry_run)

    return schedule


def install_repredict_cron(schedule: dict, dry_run: bool = False):
    """Install per-game-day cron entries for pregame_repredict.py.

    Fires ~90 min before each game day's first kickoff.
    """
    repredict_times = schedule.get("repredict_times", {})
    if not repredict_times:
        return

    try:
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, check=True,
        )
        current = result.stdout
    except subprocess.CalledProcessError:
        print("  ⚠ No existing crontab — skipping repredict cron")
        return

    # Remove old repredict entries
    lines = current.split("\n")
    cleaned = [l for l in lines if "nrl-cron.sh repredict" not in l
               and "Pre-kickoff repredict" not in l]

    # Add new repredict entries (one per game day)
    new_entries = ["# Pre-kickoff repredict (auto-set by plan_week.py)"]
    for day_key, info in sorted(repredict_times.items()):
        dt = datetime.strptime(day_key, "%Y-%m-%d")
        py_dow = dt.weekday()
        cron_dow = (py_dow + 1) % 7
        minute = info["minute"]
        hour = info["hour"]
        new_entries.append(
            f"{minute} {hour} * * {cron_dow} "
            f"/root/NRL-Predict/scripts/nrl-cron.sh repredict"
            f"  # {info['day_name']} {day_key} (90min before {info['first_kickoff']})"
        )

    # Insert before the esac/end marker or at the end
    cleaned.extend(new_entries)
    new_crontab = "\n".join(cleaned)
    if not new_crontab.endswith("\n"):
        new_crontab += "\n"

    if dry_run:
        print(f"\n  Would install {len(repredict_times)} repredict cron entries:")
        for entry in new_entries[1:]:
            print(f"    {entry}")
        return

    proc = subprocess.run(
        ["crontab", "-"], input=new_crontab, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"  ✗ Failed to install repredict cron: {proc.stderr}")
        return

    print(f"  ✓ Installed {len(repredict_times)} repredict cron entries:")
    for day_key, info in sorted(repredict_times.items()):
        print(f"    {info['day_name']:10s} {info['time']} → repredict "
              f"(first KO {info['first_kickoff']})")


if __name__ == "__main__":
    main()
