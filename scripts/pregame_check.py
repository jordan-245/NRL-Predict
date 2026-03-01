#!/usr/bin/env python3
"""
NRL Pre-Game Check: Late Odds Refresh + Lineup Monitor
======================================================
Runs periodically on game days (Thu–Sun). For each game kicking off in
the next 30–90 minutes:
  1. Fetches fresh odds from The Odds API
  2. Re-blends model prediction with fresh odds
  3. For close games (<60% confidence): if tip flips, re-submits
  4. Fetches NRL.com team lists and reports scratches to Telegram
  5. Alerts on missing tips

Late odds refresh (backtested 2018-2025, 1246 matches):
  - +13 tips improvement, 0 years negative, 55% flip accuracy
  - Market moves toward winner 61.7% of the time

Lineup impact adjustments are disabled (backtested as noise).
Lineup changes are reported for manual review only.

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

# ── Late odds refresh policy ──────────────────────────────────────
# Backtested over 2018-2025 (1246 matches):
#   - Closing odds beat opening odds by +13 tips (0 years negative)
#   - Market moves toward winner 61.7% of the time
#   - Flip accuracy: 55% (vs 53% for lineup impact)
#
# Strategy: for close games (blend confidence < REFRESH_THRESHOLD),
# re-blend with fresh odds before kickoff.  If the tip flips,
# auto-resubmit.  LOCKs are never touched.
ODDS_REFRESH_ENABLED = True
REFRESH_THRESHOLD = 0.60   # refresh games where open prob is within 40-60%

# Model blend weights (must match predict_round.py)
BLEND_MODEL_WEIGHT = 0.495
BLEND_ODDS_WEIGHT = 0.505

# ── Lineup impact adjustment policy ──────────────────────────────
# Backtested 2020-2025: impact adjustments are noise (direction rate
# 46.8%).  Lineup changes are reported via Telegram for info only.
AUTO_ADJUST_ENABLED = False

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


def fetch_fresh_odds() -> dict[str, dict]:
    """Fetch fresh odds from The Odds API for NRL matches.

    Costs 1 API credit. Returns dict keyed by canonical home team name:
    {team: {"h2h_home": float, "h2h_away": float, "home_prob": float, ...}}
    """
    try:
        from scraping.odds_api import get_odds

        raw_events = get_odds(regions="au", markets="h2h")
        if not raw_events:
            return {}

        result = {}
        for ev in raw_events:
            home_raw = ev.get("home_team", "")
            away_raw = ev.get("away_team", "")

            try:
                home = standardise_team_name(home_raw)
                away = standardise_team_name(away_raw)
            except KeyError:
                continue

            # Extract best available h2h odds (average across bookmakers)
            h2h_home_prices = []
            h2h_away_prices = []
            for bm in ev.get("bookmakers", []):
                for market in bm.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        try:
                            ot = standardise_team_name(outcome["name"])
                        except KeyError:
                            continue
                        if ot == home:
                            h2h_home_prices.append(outcome["price"])
                        elif ot == away:
                            h2h_away_prices.append(outcome["price"])

            if not h2h_home_prices or not h2h_away_prices:
                continue

            # Use median for robustness
            import statistics
            h2h_home = statistics.median(h2h_home_prices)
            h2h_away = statistics.median(h2h_away_prices)

            if h2h_home <= 0 or h2h_away <= 0:
                continue

            home_prob = (1 / h2h_home) / (1 / h2h_home + 1 / h2h_away)
            result[home] = {
                "home_team": home,
                "away_team": away,
                "h2h_home": round(h2h_home, 3),
                "h2h_away": round(h2h_away, 3),
                "home_prob": home_prob,
                "bookmakers": len(h2h_home_prices),
            }
        return result
    except Exception as e:
        log(f"  ⚠ Odds API error: {e}")
        return {}


def check_odds_refresh(
    event: dict,
    predictions: list[dict],
    fresh_odds: dict[str, dict],
) -> dict | None:
    """Check if fresh odds warrant a tip change for a close game.

    Returns dict with refresh info, or None if no action needed.
    """
    comps = event.get("competitors", [])
    home_comp = next((c for c in comps if c.get("homeAway") == "home"), comps[0])
    away_comp = next((c for c in comps if c.get("homeAway") == "away"), comps[1])
    api_home = TEAM_NAME_BY_ID.get(home_comp["teamId"], "?")
    api_away = TEAM_NAME_BY_ID.get(away_comp["teamId"], "?")

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
        return None

    old_prob = pred["home_win_prob"]
    old_tip = pred["tip"]
    try:
        old_tip = standardise_team_name(old_tip)
    except KeyError:
        pass

    # Is this a close game? (below REFRESH_THRESHOLD)
    old_confidence = abs(old_prob - 0.5) * 2
    if old_confidence > (REFRESH_THRESHOLD - 0.5) * 2:
        return None  # LOCK — don't touch

    # Find fresh odds for this match
    fresh = fresh_odds.get(api_home)
    if not fresh:
        # Try away team as key (in case home/away are swapped)
        fresh = fresh_odds.get(api_away)
        if not fresh:
            return None

    fresh_home_prob = fresh["home_prob"]
    # Ensure correct orientation
    if fresh.get("home_team") == api_away:
        fresh_home_prob = 1.0 - fresh_home_prob

    # Get the model component from the original prediction
    # Original blend: home_win_prob = 0.495 * model + 0.505 * old_odds_prob
    old_odds_prob = pred.get("odds_home_prob", old_prob)
    if abs(BLEND_ODDS_WEIGHT) > 0.001:
        model_pred = (old_prob - BLEND_ODDS_WEIGHT * old_odds_prob) / BLEND_MODEL_WEIGHT
        model_pred = np.clip(model_pred, 0.01, 0.99)
    else:
        model_pred = old_prob

    # Re-blend with fresh odds
    new_prob = BLEND_MODEL_WEIGHT * model_pred + BLEND_ODDS_WEIGHT * fresh_home_prob
    new_prob = np.clip(new_prob, 0.01, 0.99)
    new_tip = api_home if new_prob >= 0.5 else api_away

    is_flip = (new_tip != old_tip)
    odds_movement = fresh_home_prob - old_odds_prob

    return {
        "home": api_home,
        "away": api_away,
        "old_prob": old_prob,
        "new_prob": new_prob,
        "old_tip": old_tip,
        "new_tip": new_tip,
        "old_odds_prob": old_odds_prob,
        "fresh_odds_prob": fresh_home_prob,
        "odds_movement": odds_movement,
        "model_pred": model_pred,
        "is_flip": is_flip,
        "h2h_home": fresh.get("h2h_home"),
        "h2h_away": fresh.get("h2h_away"),
    }


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

    # Fetch NRL.com team lists for this round (fresh, no cache)
    from scraping.nrl_teamlists import (
        fetch_round_teamlists, load_baseline, diff_against_baseline,
        diff_lineups, get_expected_starters,
    )
    year = datetime.now().year
    round_num = event.get("round", 1)
    teamlists = fetch_round_teamlists(year, round_num, use_cache=False, delay=0.3)

    # Build lookup: team → current player list
    current_by_team = {}
    for tl in teamlists:
        current_by_team[tl["home_team"]] = tl["home_players"]
        current_by_team[tl["away_team"]] = tl["away_players"]

    # Load the baseline teamlist (captured during Tuesday tips pipeline).
    # This is the correct reference for detecting game-day scratches.
    baseline = load_baseline(year, round_num)
    use_baseline = baseline is not None

    # Fallback: if no baseline exists, use historical appearances (legacy).
    appearances_df = None
    if not use_baseline:
        app_path = PROJECT_ROOT / "data" / "processed" / "player_appearances.parquet"
        if app_path.exists():
            appearances_df = pd.read_parquet(app_path)

    home_adj = 0.0
    away_adj = 0.0
    all_changes = []

    for team, side in [(api_home, "home"), (api_away, "away")]:
        current_players = current_by_team.get(team, [])
        if not current_players:
            continue

        if use_baseline:
            # Compare against Tuesday-announced squad (correct baseline)
            changes = diff_against_baseline(team, current_players, baseline)
        else:
            # Fallback: compare against historical appearances (legacy)
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

    # Resolve old tip name
    try:
        old_tip_std = standardise_team_name(pred["tip"])
    except KeyError:
        old_tip_std = pred["tip"]

    # ── Adjustment policy ──────────────────────────────────────
    # Backtested 2020-2025: impact adjustments are anti-predictive
    # (direction rate 46.8%, swing accuracy 53.2%).  Default is
    # info-only: report scratches but do NOT adjust probs or tips.
    if AUTO_ADJUST_ENABLED:
        MAX_TEAM_ADJ = 0.15
        MAX_NET_ADJ = 0.15
        home_adj = np.clip(home_adj, -MAX_TEAM_ADJ, MAX_TEAM_ADJ)
        away_adj = np.clip(away_adj, -MAX_TEAM_ADJ, MAX_TEAM_ADJ)
        total_adj = np.clip(home_adj - away_adj, -MAX_NET_ADJ, MAX_NET_ADJ)
        if round_num <= 1:
            total_adj *= 0.25

        new_prob = np.clip(old_prob + total_adj, 0.05, 0.95)
        new_tip = api_home if new_prob >= 0.5 else api_away

        # Swing guard: block swings on strong favourites
        old_confidence = abs(old_prob - 0.5) * 2
        if new_tip != old_tip_std and old_confidence > 0.20:
            new_tip = old_tip_std
    else:
        # Info-only: compute theoretical shift for display, never change tip
        MAX_TEAM_ADJ = 0.15
        MAX_NET_ADJ = 0.15
        h = np.clip(home_adj, -MAX_TEAM_ADJ, MAX_TEAM_ADJ)
        a = np.clip(away_adj, -MAX_TEAM_ADJ, MAX_TEAM_ADJ)
        total_adj = np.clip(h - a, -MAX_NET_ADJ, MAX_NET_ADJ)
        new_prob = np.clip(old_prob + total_adj, 0.05, 0.95)
        new_tip = old_tip_std  # never change the tip

    result["lineup_changes"] = all_changes
    result["prob_shift"] = new_prob - old_prob
    result["new_prob"] = new_prob
    result["new_tip"] = new_tip
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

def _format_change_lines(changes: list[dict], max_lines: int = 8) -> list[str]:
    """Format lineup changes into readable Telegram lines.

    Shows all changes sorted by impact, with position context.
    """
    if not changes:
        return []

    # Sort: highest absolute impact first, then by jersey number
    sorted_changes = sorted(
        changes,
        key=lambda c: (-abs(c.get("impact", 0)), c.get("jersey_number", 99)),
    )

    lines = []
    for c in sorted_changes[:max_lines]:
        imp = c.get("impact", 0)
        team_short = c.get("team", "").split()[-1]
        expected = _esc(c.get("expected", "?"))
        actual = c.get("actual")
        jersey = c.get("jersey_number", "?")
        change_type = c.get("change_type", "?")

        if change_type == "MISSING":
            replacement = "OUT"
        else:
            replacement = _esc(actual) if actual else "?"

        # Impact indicator
        if abs(imp) > 0.05:
            emoji = "🔴"  # high-impact scratch
        elif abs(imp) > 0.01:
            emoji = "🟡"  # moderate
        else:
            emoji = "🔵"  # low/unknown impact

        lines.append(
            f"  {emoji} {_esc(team_short)} #{jersey}: "
            f"{expected} → {replacement}"
        )

    if len(sorted_changes) > max_lines:
        lines.append(f"  … and {len(sorted_changes) - max_lines} more")

    return lines


def send_pregame_report(
    checks: list[dict],
    swings: list[dict],
    now: datetime,
    odds_refreshes: list[dict] | None = None,
    odds_flips: list[dict] | None = None,
) -> bool:
    """Send pre-game check summary to Telegram."""
    ts = now.astimezone(AEST).strftime("%H:%M AEST")
    lines = [f"📋 <b>Pre-Game Check</b> — {ts}", ""]

    odds_refreshes = odds_refreshes or []
    odds_flips = odds_flips or []

    # Build set of games that had odds flips (to annotate in the lineup section)
    odds_flip_games = {(r["home"], r["away"]) for r in odds_flips}

    # Group info by game
    game_keys = []
    for chk in checks:
        key = (chk["home"], chk["away"])
        if key not in game_keys:
            game_keys.append(key)

    for home, away in game_keys:
        home_short = home.split()[-1]
        away_short = away.split()[-1]

        # Find matching check and odds refresh
        chk = next((c for c in checks if c["home"] == home and c["away"] == away), None)
        oref = next((r for r in odds_refreshes if r["home"] == home and r["away"] == away), None)

        ko = chk["event"]["_kickoff"].astimezone(AEST).strftime("%I:%M%p") if chk else "?"
        n_changes = len(chk["lineup_changes"]) if chk else 0

        # ── Odds refresh line ──
        if oref and oref["is_flip"]:
            old_short = _esc(oref["old_tip"].split()[-1])
            new_short = _esc(oref["new_tip"].split()[-1])
            move = oref["odds_movement"]
            lines.append(
                f"📊 <b>{_esc(home_short)} v {_esc(away_short)}</b> ({ko})"
                f"\n   Odds shifted {move:+.1%} → TIP CHANGED: "
                f"{old_short} → <b>{new_short}</b>"
                f"\n   Odds: ${oref['h2h_home']:.2f} / ${oref['h2h_away']:.2f}"
            )
        elif oref and abs(oref["odds_movement"]) > 0.02:
            move = oref["odds_movement"]
            tip_short = _esc(oref["old_tip"].split()[-1])
            lines.append(
                f"📊 <b>{_esc(home_short)} v {_esc(away_short)}</b> ({ko})"
                f"\n   Odds drifted {move:+.1%} — tip holds ({tip_short})"
                f"\n   Odds: ${oref['h2h_home']:.2f} / ${oref['h2h_away']:.2f}"
            )
        else:
            lines.append(
                f"✅ <b>{_esc(home_short)} v {_esc(away_short)}</b> ({ko})"
            )
            if oref:
                lines.append(f"   Odds: ${oref['h2h_home']:.2f} / ${oref['h2h_away']:.2f}")

        # ── Lineup changes line ──
        if n_changes > 0:
            lines.append(f"   {n_changes} late scratch{'es' if n_changes != 1 else ''}:")
            lines.extend(_format_change_lines(chk["lineup_changes"]))
        elif chk:
            lines.append(f"   No lineup changes")

        lines.append("")

    if not checks and not odds_refreshes:
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

    status = "✅ Re-submitted" if success else "❌ FAILED to re-submit"

    lines = [
        f"⚡ <b>TIP SWING — {_esc(home_short)} v {_esc(away_short)}</b>",
        f"<i>{ts} — kickoff {ko}</i>",
        "",
        f"Old tip: {_esc(old_tip_short)}",
        f"New tip: <b>{_esc(new_tip_short)}</b> ({chk['prob_shift']:+.1%} shift)",
        "",
    ]
    change_lines = _format_change_lines(chk["lineup_changes"])
    if change_lines:
        lines.append("Late changes:")
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
        log("✗ No predictions file — cannot check")
        send_message(
            f"🚨 <b>NRL Pre-Game Check FAILED</b>\n\n"
            f"No predictions file for Round {round_num}. "
            f"Tuesday tipping job may not have run."
        )
        sys.exit(1)

    # ── Step 1: Late odds refresh ─────────────────────────────────
    odds_refreshes = []
    odds_flips = []
    if ODDS_REFRESH_ENABLED:
        log("Fetching fresh odds...")
        fresh_odds = fetch_fresh_odds()
        if fresh_odds:
            log(f"  Got odds for {len(fresh_odds)} matches")
            for event in in_window:
                refresh = check_odds_refresh(event, predictions, fresh_odds)
                if refresh:
                    odds_refreshes.append({"event": event, **refresh})
                    if refresh["is_flip"]:
                        odds_flips.append({"event": event, **refresh})
                        log(f"  📊 ODDS FLIP: {refresh['old_tip'].split()[-1]} → "
                            f"{refresh['new_tip'].split()[-1]} "
                            f"(odds {refresh['old_odds_prob']:.1%} → {refresh['fresh_odds_prob']:.1%})")
                        # Auto-resubmit
                        ok = resubmit_tip(event, refresh["new_tip"], round_num, headers)
                        log(f"  {'✓' if ok else '✗'} Re-submitted: {refresh['new_tip']}")
                    else:
                        move = refresh["odds_movement"]
                        if abs(move) > 0.02:
                            log(f"  📊 Odds moved {move:+.1%} for "
                                f"{refresh['home'].split()[-1]} v {refresh['away'].split()[-1]}"
                                f" — tip holds")
        else:
            log("  No fresh odds available")

    # ── Step 2: Lineup check (info only) ──────────────────────────
    checks = []
    for event in in_window:
        comps = event.get("competitors", [])
        names = " v ".join(c.get("name", "?") for c in comps)
        log(f"Checking lineups: {names}...")

        chk = run_lineup_check_for_game(event, predictions, headers)
        checks.append(chk)

        n = len(chk["lineup_changes"])
        if n > 0:
            log(f"  {n} lineup changes (info only)")

    # ── Step 3: Send Telegram report ──────────────────────────────
    if checks or odds_refreshes:
        send_pregame_report(
            checks, [],  # no lineup swings (disabled)
            now,
            odds_refreshes=odds_refreshes,
            odds_flips=odds_flips,
        )

    n_flips = len(odds_flips)
    log(f"Done — {len(checks)} games checked, {n_flips} odds flips")


if __name__ == "__main__":
    main()
