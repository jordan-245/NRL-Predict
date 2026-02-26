#!/usr/bin/env python3
"""
ESPN Footytips Auto-Submit
==========================
Reads model predictions and submits tips to ESPN Footytips via their REST API.

Handles both personal tips and competition-specific tips.

Usage:
    # First time: authenticate and save token
    python scripts/footytips_submit.py --auth

    # Check API status and readiness
    python scripts/footytips_submit.py --status

    # Submit tips for current round
    python scripts/footytips_submit.py --round 1

    # Dry-run (show what would be submitted without actually posting)
    python scripts/footytips_submit.py --round 1 --dry-run

    # Submit only game 1
    python scripts/footytips_submit.py --round 1 --game 1

API notes:
    - Events endpoint: /sports/{sport}/leagues/{league}/events/game-types/tipping/rounds/{round}
    - Tips submit:     /games/sports/{sport}/leagues/{league}/game-types/tipping/rounds/{round}
    - Personal tips auto-propagate to all competitions (comp endpoint is admin-only)
    - clientId in payload is 0 (desktop) or 1 (mobile), NOT the OAuth client ID
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.team_mappings import standardise_team_name

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
API_BASE = "https://api.footytips.espn.com.au"
SPORT = "rugby-league"
LEAGUE = "nrl"
GAME_TYPE = "tipping"
CLIENT_ID = 0  # 0=desktop, 1=mobile (viewport flag, not OAuth client ID)
AFFILIATE_ID = "1"
COMPETITION_ID = "656543"  # Your competition

# Credentials file (gitignored)
CREDS_FILE = PROJECT_ROOT / "config" / ".footytips_creds.json"

# ---------------------------------------------------------------------------
# ESPN Footytips teamId mapping
# Canonical team name -> ESPN Footytips teamId
# ---------------------------------------------------------------------------
TEAM_ID_MAP: dict[str, int] = {
    "Brisbane Broncos": 1,
    "Canterbury Bulldogs": 2,
    "Canberra Raiders": 3,
    "Melbourne Storm": 4,
    "Newcastle Knights": 5,
    "Manly Sea Eagles": 6,
    "North Queensland Cowboys": 7,
    "Parramatta Eels": 8,
    "Penrith Panthers": 9,
    "Sydney Roosters": 10,
    "Cronulla Sharks": 11,
    "St George Illawarra Dragons": 12,
    "New Zealand Warriors": 14,
    "Wests Tigers": 15,
    "South Sydney Rabbitohs": 45,
    "Gold Coast Titans": 509,
    "Dolphins": 1706,
}

# Reverse lookup for display
TEAM_NAME_BY_ID = {v: k for k, v in TEAM_ID_MAP.items()}


# ---------------------------------------------------------------------------
# Credential management
# ---------------------------------------------------------------------------

def load_creds() -> dict:
    """Load stored credentials."""
    if CREDS_FILE.exists():
        return json.loads(CREDS_FILE.read_text())
    # Fall back to environment variables
    token = os.environ.get("FOOTYTIPS_TOKEN", "")
    user_id = os.environ.get("FOOTYTIPS_USER_ID", "")
    if token and user_id:
        return {"access_token": token, "user_id": user_id}
    return {}


def save_creds(access_token: str, user_id: str, swid: str = "") -> None:
    """Save credentials to disk (gitignored)."""
    CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "access_token": access_token,
        "user_id": user_id,  # numeric ESPN member ID
        "saved_at": datetime.now().isoformat(),
    }
    if swid:
        data["swid"] = swid  # Disney SWID (for reference)
    CREDS_FILE.write_text(json.dumps(data, indent=2))
    CREDS_FILE.chmod(0o600)
    print(f"✓ Credentials saved to {CREDS_FILE}")


def resolve_member_id(headers: dict) -> str | None:
    """Discover our numeric ESPN member ID from the competition members list."""
    try:
        resp = requests.get(
            f"{API_BASE}/competitions/{COMPETITION_ID}/members",
            headers=headers,
            params={"resultsLimit": "100", "t": str(int(time.time() * 1000))},
            timeout=15,
        )
        resp.raise_for_status()
        for user in resp.json().get("users", []):
            if user.get("currentUser"):
                return str(user["userId"])
    except Exception as e:
        print(f"  ⚠ Could not resolve member ID: {e}")
    return None


def get_auth_header(creds: dict) -> dict:
    """Build Authorization header matching browser format."""
    token = creds.get("access_token", "")
    if not token:
        print("✗ No access token found. Run with --auth first.")
        sys.exit(1)
    return {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "accept": "application/json, text/plain, */*",
        "origin": "https://footytips.espn.com.au",
        "referer": "https://footytips.espn.com.au/",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
        ),
    }


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def get_round_events(round_num: int, headers: dict) -> dict:
    """Fetch round events (matches) from the events endpoint."""
    url = f"{API_BASE}/sports/{SPORT}/leagues/{LEAGUE}/events/game-types/{GAME_TYPE}/rounds/{round_num}"
    params = {"t": str(int(time.time() * 1000))}
    resp = requests.get(url, headers=headers, params=params, timeout=30)

    if resp.status_code == 401:
        print("✗ Authentication failed (401). Token may have expired.")
        print("  Run: python scripts/footytips_submit.py --auth")
        sys.exit(1)

    resp.raise_for_status()
    return resp.json()


def get_round_data(round_num: int, headers: dict) -> dict:
    """Fetch round tips/scores from the tipping endpoint."""
    url = f"{API_BASE}/games/sports/{SPORT}/leagues/{LEAGUE}/game-types/{GAME_TYPE}/rounds/{round_num}"
    params = {"t": str(int(time.time() * 1000))}
    resp = requests.get(url, headers=headers, params=params, timeout=30)

    if resp.status_code == 401:
        print("✗ Authentication failed (401). Token may have expired.")
        print("  Run: python scripts/footytips_submit.py --auth")
        sys.exit(1)

    resp.raise_for_status()
    return resp.json()


def submit_tips(round_num: int, tips: list[dict], headers: dict) -> dict:
    """Submit tips to the personal tipping endpoint.

    Personal tips automatically propagate to all competitions the user
    has joined, so a separate competition POST is not needed (and that
    endpoint is admin-only anyway).

    Payload format (matches browser wire format):
        {"clientId": 0, "tips": [{"eventId": 53105, "teamId": 7, "tipMargin": 10}]}
    Tip objects only need eventId + teamId + optional tipMargin.
    """
    url = f"{API_BASE}/games/sports/{SPORT}/leagues/{LEAGUE}/game-types/{GAME_TYPE}/rounds/{round_num}"
    payload = {
        "clientId": CLIENT_ID,
        "tips": tips,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

def load_predictions(round_num: int, season: int = 2026) -> list[dict]:
    """Load model predictions from CSV output."""
    pred_file = PROJECT_ROOT / "outputs" / "predictions" / f"round_{round_num}_{season}.csv"
    if not pred_file.exists():
        print(f"✗ Prediction file not found: {pred_file}")
        print(f"  Run the model first: python predict_round.py --round {round_num}")
        sys.exit(1)

    predictions = []
    with open(pred_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append({
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "tip": row["tip"],
                "confidence": float(row["confidence"]),
                "home_win_prob": float(row["home_win_prob"]),
                "away_win_prob": float(row["away_win_prob"]),
            })
    return predictions


# ---------------------------------------------------------------------------
# Matching predictions to API events
# ---------------------------------------------------------------------------

def match_predictions_to_events(
    predictions: list[dict], events: list[dict]
) -> list[tuple[dict, dict]]:
    """Match model predictions to API event data.

    Returns list of (prediction, event) tuples.
    """
    matched = []

    for event in events:
        competitors = event.get("competitors", [])
        if len(competitors) < 2:
            continue

        # Get home/away from API
        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        api_home_id = home_comp.get("teamId")
        api_away_id = away_comp.get("teamId")

        api_home_name = TEAM_NAME_BY_ID.get(api_home_id, f"Unknown({api_home_id})")
        api_away_name = TEAM_NAME_BY_ID.get(api_away_id, f"Unknown({api_away_id})")

        # Find matching prediction
        for pred in predictions:
            try:
                pred_home = standardise_team_name(pred["home_team"])
                pred_away = standardise_team_name(pred["away_team"])
            except KeyError:
                continue

            if pred_home == api_home_name and pred_away == api_away_name:
                matched.append((pred, event))
                break
            # Sometimes home/away might be swapped
            if pred_home == api_away_name and pred_away == api_home_name:
                matched.append((pred, event))
                break
        else:
            print(f"  ⚠ No prediction found for: {api_home_name} vs {api_away_name}")

    return matched


def build_tips(
    matched: list[tuple[dict, dict]]
) -> tuple[list[dict], int | None, str | None]:
    """Build the tips array for API submission.

    Returns list of tip dicts.
    """
    tips = []

    for pred, event in matched:
        event_id = event["eventId"]
        event_status = event.get("eventStatus", "")

        if event_status.lower() != "pre":
            print(f"  ⏭ Skipping locked/started event {event_id} (status={event_status})")
            continue

        # Resolve predicted winner to teamId
        try:
            tip_team = standardise_team_name(pred["tip"])
        except KeyError:
            print(f"  ✗ Cannot resolve team name: {pred['tip']}")
            continue

        team_id = TEAM_ID_MAP.get(tip_team)
        if team_id is None:
            print(f"  ✗ No ESPN teamId for: {tip_team}")
            continue

        # Minimal tip format: only eventId + teamId + optional tipMargin
        tip: dict = {
            "eventId": event_id,
            "teamId": team_id,
        }

        # Add margin if required by the event
        if event.get("marginRequired") and event.get("margin"):
            margin_default = event["margin"].get("default", 1)
            tip["tipMargin"] = margin_default

        tips.append(tip)

    return tips


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_tips(tips: list[dict], matched: list[tuple[dict, dict]]) -> None:
    """Pretty-print the tips that will be submitted."""
    print("\n" + "=" * 60)
    print("  TIPS TO SUBMIT")
    print("=" * 60)

    event_pred_map = {event["eventId"]: pred for pred, event in matched}

    for tip in tips:
        event_id = tip["eventId"]
        team_name = TEAM_NAME_BY_ID.get(tip["teamId"], f"ID:{tip['teamId']}")
        pred = event_pred_map.get(event_id, {})
        conf = pred.get("confidence", 0)

        # Determine category
        if conf >= 0.15:
            cat = "🔒 LOCK"
        elif conf >= 0.05:
            cat = "📐 LEAN"
        else:
            cat = "🎲 TOSS-UP"

        home = pred.get("home_team", "?")
        away = pred.get("away_team", "?")
        print(f"  {cat:12s}  {home} vs {away}  →  {team_name}")

    print("=" * 60)
    print(f"  Total tips: {len(tips)}")
    print()


# ---------------------------------------------------------------------------
# Auth flow
# ---------------------------------------------------------------------------

def interactive_auth():
    """Guide user through authentication."""
    print("\n" + "=" * 60)
    print("  ESPN FOOTYTIPS AUTHENTICATION")
    print("=" * 60)
    print()
    print("To get your access token:")
    print("  1. Open https://footytips.espn.com.au/tipping/nrl in Chrome")
    print("  2. Log in to your ESPN account")
    print("  3. Open DevTools (F12) → Network tab")
    print("  4. Look for any request to api.footytips.espn.com.au")
    print("  5. Copy the 'Authorization' header value (after 'Bearer ')")
    print()

    token = input("Paste your access_token: ").strip()
    if token.startswith("Bearer "):
        token = token[7:]

    if not token:
        print("✗ Access token is required.")
        sys.exit(1)

    # Validate by making a test API call
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "NRL-Predict/1.0",
    }

    print("\nValidating credentials...")
    try:
        resp = requests.get(
            f"{API_BASE}/clients/{AFFILIATE_ID}/sports/leagues",
            headers=headers,
            params={"includeGameTypes": "true"},
            timeout=15,
        )
        if resp.status_code == 401:
            print("✗ Token is invalid or expired. Please try again.")
            sys.exit(1)
        elif resp.status_code != 200:
            print(f"⚠ Unexpected status {resp.status_code}")
    except Exception as e:
        print(f"⚠ Could not validate ({e})")

    print("✓ Token is valid!")

    # Auto-discover numeric member ID from competition
    print("\n🔍 Discovering your ESPN member ID...")
    member_id = resolve_member_id(headers)
    if member_id:
        print(f"   ✓ Found member ID: {member_id}")
    else:
        print("   ⚠ Could not auto-discover member ID.")
        member_id = input("   Enter your numeric ESPN member ID (or press Enter to skip): ").strip()
        if not member_id:
            print("   ⚠ Competition tips will not work without a member ID.")
            member_id = ""

    # Extract SWID from JWT for reference
    swid = ""
    try:
        import base64
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        jwt_data = json.loads(base64.urlsafe_b64decode(payload))
        swid = jwt_data.get("sub", "")
    except Exception:
        pass

    save_creds(token, member_id, swid=swid)


def check_status(headers: dict) -> None:
    """Check NRL tipping status and readiness."""
    print("\n" + "=" * 60)
    print("  ESPN FOOTYTIPS STATUS CHECK")
    print("=" * 60)

    # Check token validity
    try:
        import base64
        creds = load_creds()
        token = creds.get("access_token", "")
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        jwt_data = json.loads(base64.urlsafe_b64decode(payload))
        exp = jwt_data.get("exp", 0)
        remaining = exp - int(time.time())
        if remaining > 0:
            print(f"  🔑 Token valid for {remaining // 3600}h {(remaining % 3600) // 60}m")
        else:
            print(f"  ❌ Token EXPIRED {abs(remaining) // 3600}h ago — run --auth")
    except Exception:
        print("  ⚠ Could not check token expiry")

    # Check user ID
    user_id = creds.get("user_id", "")
    if user_id and user_id.isdigit():
        print(f"  👤 Member ID: {user_id}")
    elif user_id:
        print(f"  ⚠ Member ID looks wrong (not numeric): {user_id}")
        print(f"     Run --auth to auto-discover the correct ID")
    else:
        print(f"  ❌ No member ID — competition tips won't work")

    # Check NRL season status
    try:
        resp = requests.get(
            f"{API_BASE}/clients/{AFFILIATE_ID}/sports/leagues",
            headers=headers,
            params={"includeGameTypes": "true"},
            timeout=15,
        )
        if resp.status_code == 200:
            for league in resp.json().get("leagues", []):
                if league.get("slug") == "nrl":
                    gt = league.get("gameTypes", [{}])[0]
                    status = gt.get("status", "?")
                    rnd = gt.get("round", "?")
                    rnd_status = gt.get("roundStatus", "?")
                    start = gt.get("roundStartDateTime", gt.get("startDateTime", "?"))
                    print(f"  🏉 NRL status: {status}")
                    print(f"     Current round: {rnd} (status: {rnd_status})")
                    print(f"     Round start: {start}")

                    # Check events availability
                    events_resp = get_round_events(rnd, headers)
                    n_events = len(events_resp.get("events", []))
                    byes = events_resp.get("byes", [])
                    tips_resp = get_round_data(rnd, headers)
                    n_round_events = tips_resp.get("scores", {}).get("roundEvents", 0)
                    print(f"     Events available: {n_events} (+ {len(byes)} byes)")
                    if n_events == 0:
                        print(f"     ⏳ No events published yet")
                    existing = len(tips_resp.get("tips", []))
                    if existing:
                        print(f"     📝 You have {existing} tips already submitted")
                    break
        elif resp.status_code == 401:
            print("  ❌ Authentication failed — token expired")
    except Exception as e:
        print(f"  ⚠ API error: {e}")

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ESPN Footytips Auto-Submit")
    parser.add_argument("--auth", action="store_true", help="Interactive authentication setup")
    parser.add_argument("--status", action="store_true", help="Check API status and readiness")
    parser.add_argument("--round", type=int, help="Round number to submit tips for")
    parser.add_argument("--season", type=int, default=2026, help="Season year")
    parser.add_argument("--dry-run", action="store_true", help="Show tips without submitting")
    parser.add_argument("--game", type=int, help="Submit only for specific game number (1-indexed)")
    args = parser.parse_args()

    if args.auth:
        interactive_auth()
        return

    # Load credentials
    creds = load_creds()
    headers = get_auth_header(creds)
    user_id = creds.get("user_id", "")

    if args.status:
        check_status(headers)
        return

    if not args.round:
        print("✗ --round is required. Usage: python scripts/footytips_submit.py --round 5")
        sys.exit(1)

    # Validate user_id format for competition submissions
    if user_id and not user_id.isdigit():
        print(f"⚠ user_id '{user_id}' is not numeric (SWID?). Auto-discovering...")
        member_id = resolve_member_id(headers)
        if member_id:
            print(f"  ✓ Found numeric member ID: {member_id}")
            user_id = member_id
            # Update creds file
            creds["swid"] = creds.get("user_id", "")
            creds["user_id"] = member_id
            CREDS_FILE.write_text(json.dumps(creds, indent=2))
            CREDS_FILE.chmod(0o600)
        else:
            print(f"  ✗ Could not resolve. Competition tips will fail.")
            print(f"    Run --auth to set up credentials properly.")

    print(f"\n📋 Loading predictions for Round {args.round}, {args.season}...")
    predictions = load_predictions(args.round, args.season)
    print(f"   Found {len(predictions)} predictions")

    print(f"\n📡 Fetching round {args.round} events from ESPN Footytips API...")
    events_data = get_round_events(args.round, headers)
    events = events_data.get("events", [])
    byes = events_data.get("byes", [])
    print(f"   Found {len(events)} events")
    if byes:
        bye_names = ", ".join(b.get("name", "?") for b in byes)
        print(f"   Byes: {bye_names}")

    # Also fetch tips/scores data for lockout info
    round_data = get_round_data(args.round, headers)
    lockout = round_data.get("lockoutDateTime", "")
    existing_tips = round_data.get("tips", [])
    if lockout:
        print(f"   Lockout: {lockout}")
    if existing_tips:
        print(f"   ⚠ You already have {len(existing_tips)} tips submitted (will overwrite)")

    print(f"\n🔗 Matching predictions to events...")
    matched = match_predictions_to_events(predictions, events)
    print(f"   Matched {len(matched)} of {len(events)} events")

    tips = build_tips(matched)
    display_tips(tips, matched)

    if not tips:
        print("✗ No tips to submit!")
        sys.exit(1)

    # Filter to single game if requested
    if args.game:
        if args.game < 1 or args.game > len(tips):
            print(f"✗ --game {args.game} out of range (1-{len(tips)})")
            sys.exit(1)
        tips = [tips[args.game - 1]]
        print(f"   Filtered to game {args.game} only")
        display_tips(tips, matched)

    if args.dry_run:
        print("🏃 DRY RUN — nothing submitted")
        return

    # Submit tips (personal endpoint — auto-propagates to all competitions)
    print(f"📤 Submitting {len(tips)} tip(s)...")
    try:
        result = submit_tips(args.round, tips, headers)
        msg = result.get("message", "OK")
        print(f"   ✓ {msg}")
    except requests.HTTPError as e:
        print(f"   ✗ Failed: {e}")
        if e.response is not None:
            print(f"     {e.response.text[:500]}")
        sys.exit(1)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
