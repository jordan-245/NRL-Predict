#!/usr/bin/env python3
"""
NRL Pre-Game Lineup Check & Tip Adjuster
=========================================
Runs periodically on game days (Thu–Sun). For each game kicking off in
the next 30–90 minutes:
  1. Fetches NRL.com team lists
  2. Computes lineup-adjusted probabilities
  3. Compares adjusted tip with the currently submitted tip on ESPN Footytips
  4. If the tip FLIPS (different winner), re-submits just that game
  5. Sends Telegram notification for any swings or missing tips

Designed to run via cron every 30 min on game days:
    */30 13-20 * * 4-7  scripts/nrl-cron.sh pregame

Exit codes:
    0 — success (even if no games in window)
    1 — error
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
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

import numpy as np
import pandas as pd
import requests

from config.team_mappings import standardise_team_name
from scripts.footytips_submit import (
    API_BASE, SPORT, LEAGUE, GAME_TYPE, CLIENT_ID,
    TEAM_ID_MAP, TEAM_NAME_BY_ID,
    load_creds, get_auth_header, get_round_events, get_round_data,
    submit_tips, load_predictions,
)
from scripts.telegram_notify import send_message, _esc

AEST = timezone(timedelta(hours=10))
# Window: check games kicking off 30–90 min from now
WINDOW_MIN_MINUTES = 30
WINDOW_MAX_MINUTES = 90
# Swing threshold: probability shift needed to trigger a re-submit
# Only re-submits if the predicted WINNER actually flips
SWING_FLIP_ONLY = True

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def log(msg: str):
    ts = datetime.now(AEST).strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}")


def get_current_round(headers: dict) -> int | None:
    """Detect current tipping round from ESPN API."""
    try:
        resp = requests.get(
            f"{API_BASE}/clients/1/sports/leagues",
            headers=headers,
            params={"includeGameTypes": "true"},
            timeout=15,
        )
        resp.raise_for_status()
        for league in resp.json().get("leagues", []):
            if league.get("slug") == "nrl":
                gt = league.get("gameTypes", [{}])[0]
                return gt.get("round")
    except Exception:
        pass
    return None


def get_submitted_tips(round_num: int, headers: dict) -> dict[int, int]:
    """Get currently submitted tips: {eventId: teamId}."""
    try:
        data = get_round_data(round_num, headers)
        tips = data.get("tips", [])
        return {t["eventId"]: t["teamId"] for t in tips if "eventId" in t and "teamId" in t}
    except Exception:
        return {}


def find_games_in_window(events: list[dict], now: datetime) -> list[dict]:
    """Find events kicking off in the 30–90 min window from `now`."""
    window_start = now + timedelta(minutes=WINDOW_MIN_MINUTES)
    window_end = now + timedelta(minutes=WINDOW_MAX_MINUTES)

    in_window = []
    for event in events:
        if event.get("eventStatus", "").lower() != "pre":
            continue
        ko_str = event.get("dateTime", "")
        if not ko_str:
            continue
        try:
            ko = datetime.fromisoformat(ko_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if window_start <= ko <= window_end:
            event["_kickoff"] = ko
            in_window.append(event)
    return in_window


def find_missing_tips(
    events: list[dict], submitted: dict[int, int], now: datetime
) -> list[dict]:
    """Find events kicking off within 90 min that have NO submitted tip."""
    cutoff = now + timedelta(minutes=90)
    missing = []
    for event in events:
        if event.get("eventStatus", "").lower() != "pre":
            continue
        ko_str = event.get("dateTime", "")
        if not ko_str:
            continue
        try:
            ko = datetime.fromisoformat(ko_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ko <= cutoff and event["eventId"] not in submitted:
            event["_kickoff"] = ko
            missing.append(event)
    return missing


def run_lineup_check_for_game(
    event: dict, predictions: list[dict], headers: dict
) -> dict:
    """Run lineup check for a single game.

    Returns dict with:
        event, home, away, old_tip_team_id, new_tip_team_id,
        lineup_changes, prob_shift, is_swing
    """
    from scraping.nrl_teamlists import (
        fetch_match_teamlist, diff_lineups, get_expected_starters,
    )
    from processing.player_impact import get_player_impact, OUTPUT_PATH as IMPACT_PATH

    comps = event.get("competitors", [])
    home_comp = next((c for c in comps if c.get("homeAway") == "home"), comps[0])
    away_comp = next((c for c in comps if c.get("homeAway") == "away"), comps[1])
    api_home = TEAM_NAME_BY_ID.get(home_comp["teamId"], "?")
    api_away = TEAM_NAME_BY_ID.get(away_comp["teamId"], "?")

    result = {
        "event": event,
        "home": api_home,
        "away": api_away,
        "home_team_id": home_comp["teamId"],
        "away_team_id": away_comp["teamId"],
        "lineup_changes": [],
        "prob_shift": 0.0,
        "old_prob": 0.5,
        "new_prob": 0.5,
        "old_tip": api_home,
        "new_tip": api_home,
        "is_swing": False,
    }

    # Find matching prediction
    pred = None
    for p in predictions:
        try:
            ph = standardise_team_name(p["home_team"])
            pa = standardise_team_name(p["away_team"])
        except KeyError:
            continue
        if (ph == api_home and pa == api_away) or (ph == api_away and pa == api_home):
            pred = p
            break

    if not pred:
        log(f"  ⚠ No prediction for {api_home} vs {api_away}")
        return result

    old_prob = pred["home_win_prob"]
    result["old_prob"] = old_prob
    result["old_tip"] = pred["tip"]

    # Load player impact data
    impact_df = None
    if IMPACT_PATH.exists():
        impact_df = pd.read_parquet(IMPACT_PATH)

    # Load appearances for expected starters
    app_path = PROJECT_ROOT / "data" / "processed" / "player_appearances.parquet"
    appearances_df = None
    if app_path.exists():
        appearances_df = pd.read_parquet(app_path)

    # Fetch NRL.com team lists for this round
    from scraping.nrl_teamlists import fetch_round_teamlists
    year = datetime.now().year
    round_num = event.get("round", 1)
    teamlists = fetch_round_teamlists(year, round_num, use_cache=False, delay=0.3)

    # Build lookup
    current_by_team = {}
    for tl in teamlists:
        current_by_team[tl["home_team"]] = tl["home_players"]
        current_by_team[tl["away_team"]] = tl["away_players"]

    home_adj = 0.0
    away_adj = 0.0
    all_changes = []

    for team, side in [(api_home, "home"), (api_away, "away")]:
        current_players = current_by_team.get(team, [])
        if not current_players:
            continue
        expected = get_expected_starters(team, appearances_df, n_recent=5)
        if not expected:
            continue
        changes = diff_lineups(team, current_players, expected)
        for change in changes:
            impact = 0.0
            if impact_df is not None:
                impact = get_player_impact(
                    team, player_name=change["expected"], impact_df=impact_df
                )
            change["impact"] = impact
            change["team"] = team
            change["side"] = side
            all_changes.append(change)
            if side == "home":
                home_adj -= impact
            else:
                away_adj -= impact

    # Cap adjustments
    MAX_ADJ = 0.15
    home_adj = np.clip(home_adj, -MAX_ADJ, MAX_ADJ)
    away_adj = np.clip(away_adj, -MAX_ADJ, MAX_ADJ)
    total_adj = home_adj - away_adj

    new_prob = np.clip(old_prob + total_adj, 0.05, 0.95)
    new_tip = api_home if new_prob >= 0.5 else api_away

    result["lineup_changes"] = all_changes
    result["prob_shift"] = new_prob - old_prob
    result["new_prob"] = new_prob
    result["new_tip"] = new_tip

    try:
        old_tip_std = standardise_team_name(pred["tip"])
    except KeyError:
        old_tip_std = pred["tip"]
    result["old_tip"] = old_tip_std
    result["is_swing"] = (new_tip != old_tip_std)

    return result


def resubmit_tip(event: dict, new_tip_team: str, round_num: int, headers: dict) -> bool:
    """Re-submit a single game's tip on ESPN Footytips."""
    team_id = TEAM_ID_MAP.get(new_tip_team)
    if not team_id:
        log(f"  ✗ No ESPN teamId for {new_tip_team}")
        return False

    tip = {"eventId": event["eventId"], "teamId": team_id}
    if event.get("marginRequired") and event.get("margin"):
        tip["tipMargin"] = event["margin"].get("default", 10)

    try:
        result = submit_tips(round_num, [tip], headers)
        msg = result.get("message", "OK")
        log(f"  ✓ Re-submitted: {new_tip_team} — {msg}")
        return True
    except Exception as e:
        log(f"  ✗ Re-submit failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Telegram messages
# ---------------------------------------------------------------------------

def send_pregame_report(checks: list[dict], swings: list[dict], now: datetime) -> bool:
    """Send pre-game lineup check summary to Telegram."""
    ts = now.astimezone(AEST).strftime("%H:%M AEST")
    lines = [f"📋 <b>Pre-Game Lineup Check</b> — {ts}", ""]

    for chk in checks:
        home_short = chk["home"].split()[-1]
        away_short = chk["away"].split()[-1]
        ko = chk["event"]["_kickoff"].astimezone(AEST).strftime("%I:%M%p")
        n_changes = len(chk["lineup_changes"])

        if chk["is_swing"]:
            old_tip_short = chk["old_tip"].split()[-1]
            new_tip_short = chk["new_tip"].split()[-1]
            shift = chk["prob_shift"]
            lines.append(
                f"⚡ <b>{_esc(home_short)} v {_esc(away_short)}</b> ({ko})"
                f"\n   {n_changes} changes → TIP CHANGED: "
                f"{_esc(old_tip_short)} → <b>{_esc(new_tip_short)}</b>"
                f" ({shift:+.1%} shift)"
            )
        elif n_changes > 0:
            lines.append(
                f"✅ <b>{_esc(home_short)} v {_esc(away_short)}</b> ({ko})"
                f"\n   {n_changes} changes — no swing, tip holds"
            )
        else:
            lines.append(
                f"✅ <b>{_esc(home_short)} v {_esc(away_short)}</b> ({ko})"
                f"\n   No lineup changes"
            )

    if not checks:
        lines.append("No games in pre-kickoff window.")

    return send_message("\n".join(lines))


def send_missing_tips_alert(missing: list[dict], now: datetime) -> bool:
    """Send urgent alert for games about to kick off without a tip."""
    ts = now.astimezone(AEST).strftime("%H:%M AEST")
    lines = [
        f"🚨 <b>MISSING TIPS — Action Required!</b>",
        f"<i>{ts}</i>",
        "",
    ]
    for event in missing:
        comps = event.get("competitors", [])
        names = " v ".join(c.get("name", "?") for c in comps)
        ko = event["_kickoff"].astimezone(AEST).strftime("%I:%M%p")
        lines.append(f"  ❌ {_esc(names)} — kicks off {ko}")

    lines.append("")
    lines.append("Tips were NOT submitted for these games!")

    return send_message("\n".join(lines))


def send_swing_alert(chk: dict, success: bool, now: datetime) -> bool:
    """Send individual swing alert when a tip is changed."""
    ts = now.astimezone(AEST).strftime("%H:%M AEST")
    home_short = chk["home"].split()[-1]
    away_short = chk["away"].split()[-1]
    old_tip_short = chk["old_tip"].split()[-1]
    new_tip_short = chk["new_tip"].split()[-1]
    ko = chk["event"]["_kickoff"].astimezone(AEST).strftime("%I:%M%p")

    # List key changes
    change_lines = []
    for c in sorted(chk["lineup_changes"], key=lambda x: abs(x.get("impact", 0)), reverse=True)[:5]:
        imp = c.get("impact", 0)
        if abs(imp) > 0.001:
            emoji = "📈" if imp > 0 else "📉"
            change_lines.append(
                f"  {emoji} {_esc(c['team'].split()[-1])} #{c['jersey_number']}: "
                f"{_esc(c['expected'])} → {_esc(c.get('actual', 'OUT'))} "
                f"({imp:+.3f})"
            )

    status = "✅ Re-submitted" if success else "❌ FAILED to re-submit"

    lines = [
        f"⚡ <b>TIP SWING — {_esc(home_short)} v {_esc(away_short)}</b>",
        f"<i>{ts} — kickoff {ko}</i>",
        "",
        f"Old tip: {_esc(old_tip_short)}",
        f"New tip: <b>{_esc(new_tip_short)}</b> ({chk['prob_shift']:+.1%} shift)",
        "",
    ]
    if change_lines:
        lines.append("Key changes:")
        lines.extend(change_lines)
        lines.append("")
    lines.append(status)

    return send_message("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    now = datetime.now(timezone.utc)
    now_aest = now.astimezone(AEST)
    log(f"Pre-game check starting ({now_aest.strftime('%a %H:%M AEST')})")

    # Load creds and check token
    creds = load_creds()
    if not creds.get("access_token"):
        log("✗ No token — run footytips_auth.py first")
        send_message("🚨 <b>NRL Pre-Game Check FAILED</b>\n\nNo ESPN token available.")
        sys.exit(1)

    headers = get_auth_header(creds)

    # Detect current round
    round_num = get_current_round(headers)
    if not round_num:
        log("Could not detect current round — skipping")
        return

    log(f"Round {round_num}")

    # Fetch events
    events_data = get_round_events(round_num, headers)
    events = events_data.get("events", [])
    if not events:
        log("No events available — skipping")
        return

    # Get submitted tips
    submitted = get_submitted_tips(round_num, headers)
    log(f"{len(submitted)}/{len(events)} tips currently submitted")

    # Check for missing tips on games approaching kickoff
    missing = find_missing_tips(events, submitted, now)
    if missing:
        log(f"⚠ {len(missing)} games approaching kickoff WITHOUT tips!")
        send_missing_tips_alert(missing, now)

    # Find games in the pre-kickoff window
    in_window = find_games_in_window(events, now)
    if not in_window:
        log("No games in 30–90 min window — done")
        return

    game_descs = []
    for e in in_window:
        comps = e.get("competitors", [])
        names = " v ".join(c.get("name", "?") for c in comps)
        ko = e["_kickoff"].astimezone(AEST).strftime("%I:%M%p")
        game_descs.append(f"{names} ({ko})")
    log(f"Games in window: {', '.join(game_descs)}")

    # Load predictions
    year = now_aest.year
    try:
        predictions = load_predictions(round_num, year)
    except SystemExit:
        log("✗ No predictions file — cannot check for swings")
        send_message(
            f"🚨 <b>NRL Pre-Game Check FAILED</b>\n\n"
            f"No predictions file for Round {round_num}. "
            f"Tuesday tipping job may not have run."
        )
        sys.exit(1)

    # Run lineup check for each game in window
    checks = []
    swings = []
    for event in in_window:
        comps = event.get("competitors", [])
        names = " v ".join(c.get("name", "?") for c in comps)
        log(f"Checking {names}...")

        chk = run_lineup_check_for_game(event, predictions, headers)
        checks.append(chk)

        n = len(chk["lineup_changes"])
        log(f"  {n} lineup changes, shift={chk['prob_shift']:+.3f}, swing={chk['is_swing']}")

        if chk["is_swing"]:
            swings.append(chk)
            log(f"  ⚡ SWING: {chk['old_tip']} → {chk['new_tip']}")
            # Re-submit the changed tip
            ok = resubmit_tip(event, chk["new_tip"], round_num, headers)
            send_swing_alert(chk, ok, now)

    # Send summary to Telegram (only if there were games to check)
    if checks and not swings:
        # No swings — send a quiet summary
        send_pregame_report(checks, swings, now)

    log(f"Done — {len(checks)} games checked, {len(swings)} swings")


if __name__ == "__main__":
    main()
