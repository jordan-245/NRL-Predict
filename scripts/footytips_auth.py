#!/usr/bin/env python3
"""
ESPN Footytips — Automated Token Refresh
==========================================
Uses Playwright (headless Chromium) to authenticate with Disney OneID
and extract a fresh access_token for the Footytips API.

The Disney OneID token expires every ~24 hours. This script:
  1. Opens footytips.espn.com.au in headless Chromium
  2. If a saved browser session exists, OneID silently refreshes the token
  3. If no session, logs in with email/password (first time or --fresh)
  4. Intercepts API calls to capture the fresh Bearer token
  5. Saves the token + member ID to config/.footytips_creds.json

Usage:
    # First time: will prompt for email/password
    python scripts/footytips_auth.py

    # Subsequent runs: silent refresh using saved browser state
    python scripts/footytips_auth.py

    # Force fresh login (clear saved browser state)
    python scripts/footytips_auth.py --fresh

    # Just check if token needs refresh
    python scripts/footytips_auth.py --check

    # Provide OTP code (Disney OneID requires OTP from new devices)
    python scripts/footytips_auth.py --fresh --otp 123456

Credentials:
    Email/password can be provided via environment variables to avoid prompts:
        FOOTYTIPS_EMAIL=you@example.com
        FOOTYTIPS_PASSWORD=yourpassword
    OTP code (if Disney OneID requires verification):
        FOOTYTIPS_OTP=123456
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Auto-load .env file
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())
CREDS_FILE = PROJECT_ROOT / "config" / ".footytips_creds.json"
BROWSER_STATE_DIR = PROJECT_ROOT / "config" / ".footytips_browser_state"
COMPETITION_ID = "656543"
API_BASE = "https://api.footytips.espn.com.au"
# ---------------------------------------------------------------------------


def check_token_expiry() -> tuple[bool, int]:
    """Returns (is_valid, seconds_remaining)."""
    if not CREDS_FILE.exists():
        return False, 0
    try:
        creds = json.loads(CREDS_FILE.read_text())
        token = creds.get("access_token", "")
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        jwt_data = json.loads(base64.urlsafe_b64decode(payload))
        remaining = jwt_data.get("exp", 0) - int(time.time())
        return remaining > 0, remaining
    except Exception:
        return False, 0


def resolve_member_id(token: str) -> str | None:
    """Look up our numeric ESPN member ID from the competition."""
    import requests
    try:
        resp = requests.get(
            f"{API_BASE}/competitions/{COMPETITION_ID}/members",
            headers={
                "authorization": f"Bearer {token}",
                "accept": "application/json",
                "origin": "https://footytips.espn.com.au",
                "referer": "https://footytips.espn.com.au/",
            },
            params={"resultsLimit": "100"},
            timeout=15,
        )
        for user in resp.json().get("users", []):
            if user.get("currentUser"):
                return str(user["userId"])
    except Exception:
        pass
    return None


def save_creds(access_token: str, member_id: str = "", swid: str = "") -> None:
    """Save credentials to config/.footytips_creds.json."""
    CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CREDS_FILE.exists():
        existing = json.loads(CREDS_FILE.read_text())
        member_id = member_id or existing.get("user_id", "")
        swid = swid or existing.get("swid", "")

    data = {"access_token": access_token, "user_id": member_id,
            "saved_at": datetime.now().isoformat()}
    if swid:
        data["swid"] = swid
    CREDS_FILE.write_text(json.dumps(data, indent=2))
    CREDS_FILE.chmod(0o600)


GMAIL_ACCOUNT = "jordanbaillie@gmail.com"


def _snapshot_gmail_otp_ids() -> tuple[set[str], str | None]:
    """Snapshot existing ESPN OTP message IDs BEFORE triggering login.
    Returns (set of message IDs, thread ID or None)."""
    old_msg_ids: set[str] = set()
    otp_thread_id: str | None = None
    try:
        result = subprocess.run(
            ["gmcli", GMAIL_ACCOUNT, "search",
             'from:support@espn.com subject:"Your ESPN Account Passcode"',
             "--max", "3"],
            capture_output=True, text=True, timeout=15,
        )
        for line in result.stdout.strip().split("\n"):
            tid = line.split("\t")[0].strip()
            if tid and tid != "ID":
                otp_thread_id = otp_thread_id or tid
        if otp_thread_id:
            result = subprocess.run(
                ["gmcli", GMAIL_ACCOUNT, "thread", otp_thread_id],
                capture_output=True, text=True, timeout=15,
            )
            for m in re.finditer(r"^Message-ID:\s*(\S+)", result.stdout, re.MULTILINE):
                old_msg_ids.add(m.group(1))
    except Exception:
        pass
    return old_msg_ids, otp_thread_id


def _poll_gmail_otp(
    timeout: int = 180,
    poll_interval: int = 10,
    pre_snapshot: tuple[set[str], str | None] | None = None,
) -> str | None:
    """Poll Gmail for the latest ESPN OTP code (skips pre-existing messages).
    If pre_snapshot is provided, uses those as the 'old' baseline instead of
    snapshotting now (avoids race condition where OTP arrives before snapshot).
    """
    print("  📧 Polling Gmail for OTP...")

    if pre_snapshot is not None:
        old_msg_ids, otp_thread_id = pre_snapshot
        print(f"  📋 Using pre-login snapshot ({len(old_msg_ids)} existing messages)")
    else:
        old_msg_ids, otp_thread_id = _snapshot_gmail_otp_ids()

    start = time.time()
    while time.time() - start < timeout:
        elapsed = int(time.time() - start)
        try:
            if not otp_thread_id:
                r = subprocess.run(
                    ["gmcli", GMAIL_ACCOUNT, "search",
                     'from:support@espn.com subject:"Your ESPN Account Passcode"',
                     "--max", "1"],
                    capture_output=True, text=True, timeout=15,
                )
                for line in r.stdout.strip().split("\n"):
                    tid = line.split("\t")[0].strip()
                    if tid and tid != "ID":
                        otp_thread_id = tid
                        break

            if otp_thread_id:
                result = subprocess.run(
                    ["gmcli", GMAIL_ACCOUNT, "thread", otp_thread_id],
                    capture_output=True, text=True, timeout=15,
                )
                messages = re.split(r"^---$", result.stdout, flags=re.MULTILINE)
                for msg in reversed(messages):
                    mid_m = re.search(r"^Message-ID:\s*(\S+)", msg, re.MULTILINE)
                    if not mid_m or mid_m.group(1) in old_msg_ids:
                        continue
                    # Found a NEW message — extract OTP code
                    code_m = re.search(r"font-weight:\s*bold[^>]*>(\d{6})<", msg)
                    if code_m:
                        print(f"  ✅ OTP from Gmail: {code_m.group(1)}")
                        return code_m.group(1)
                    codes = re.findall(r'\b(\d{6})\b', msg)
                    if codes:
                        print(f"  ✅ OTP from Gmail: {codes[-1]}")
                        return codes[-1]
        except Exception as e:
            print(f"  ⚠ Gmail error: {e}")

        print(f"  ⏳ Waiting for OTP email... ({elapsed}s)")
        time.sleep(poll_interval)

    return None


def _find_oneid_frame(page):
    """Locate the OneID iframe within the page."""
    for frame in page.frames:
        if "oneid-iframe" in (frame.name or "") or "registerdisney" in frame.url:
            return frame
    return None


def _detect_otp_screen(target) -> bool:
    """Check if the OTP verification screen is showing."""
    try:
        otp_inputs = target.locator(
            "input[type='tel'], input[inputmode='numeric']"
        )
        if otp_inputs.count() >= 6:
            return True
        heading = target.locator("h1, h2")
        for i in range(heading.count()):
            txt = heading.nth(i).text_content().strip().lower()
            if "check your email" in txt or "enter" in txt and "code" in txt:
                return True
    except Exception:
        pass
    return False


def _enter_otp(target, page, otp_code: str) -> None:
    """Fill the 6-digit OTP code into the OneID verification form."""
    print(f"  🔢 Entering OTP code...")
    otp_inputs = target.locator("input[type='tel'], input[inputmode='numeric']")
    count = otp_inputs.count()
    if count >= 6 and len(otp_code) == 6:
        # Fill each digit into its own input field
        for i, digit in enumerate(otp_code):
            otp_inputs.nth(i).fill(digit)
            page.wait_for_timeout(100)
    elif count == 1:
        # Single input field for full code
        otp_inputs.first.fill(otp_code)
    else:
        # Try typing into the first field (some forms auto-advance)
        otp_inputs.first.click()
        page.keyboard.type(otp_code, delay=100)

    # Click Continue / Submit
    page.wait_for_timeout(500)
    try:
        target.click("#BtnSubmit", timeout=3000)
    except Exception:
        try:
            target.click('button:has-text("Continue")', timeout=3000)
        except Exception:
            pass


def _extract_token_fallbacks(page, ctx) -> str | None:
    """Try localStorage, Redux store, and cookies to find a JWT token.
    Only returns tokens that are still valid (≥5 min remaining)."""
    # localStorage
    try:
        token = page.evaluate("""() => {
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                const val = localStorage.getItem(key);
                if (val && val.startsWith('ey') && val.includes('.')) {
                    const parts = val.split('.');
                    if (parts.length === 3) return val;
                }
                if (val && val.includes('"access_token"')) {
                    try {
                        const obj = JSON.parse(val);
                        if (obj.access_token && obj.access_token.startsWith('ey'))
                            return obj.access_token;
                    } catch {}
                }
            }
            return null;
        }""")
        if token and _is_token_fresh(token):
            print("  ✓ Token extracted from localStorage")
            return token
    except Exception:
        pass

    # Redux store
    try:
        token = page.evaluate("""() => {
            try {
                const store = window.__REDUX_STORE__ || window.__store__;
                if (store) {
                    const state = store.getState();
                    const auth = state?.auth?.authorization || '';
                    if (auth.startsWith('Bearer ey')) return auth.slice(7);
                }
            } catch {}
            return null;
        }""")
        if token and _is_token_fresh(token):
            print("  ✓ Token extracted from Redux store")
            return token
    except Exception:
        pass

    # Cookies
    try:
        for c in ctx.cookies():
            if (c["value"].startswith("ey") and c["value"].count(".") == 2
                    and _is_token_fresh(c["value"])):
                print(f"  ✓ Token extracted from cookie: {c['name']}")
                return c["value"]
    except Exception:
        pass

    return None


def _is_token_fresh(token: str, min_remaining: int = 300) -> bool:
    """Check if a JWT token has at least min_remaining seconds of validity."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        jwt_data = json.loads(base64.urlsafe_b64decode(payload))
        remaining = jwt_data.get("exp", 0) - int(time.time())
        return remaining >= min_remaining
    except Exception:
        return False


def refresh_token(fresh: bool = False, otp_code: str = "") -> str | None:
    """Use Playwright to get a fresh token from ESPN Footytips."""
    from playwright.sync_api import sync_playwright
    import shutil

    state_dir = str(BROWSER_STATE_DIR)
    if fresh and BROWSER_STATE_DIR.exists():
        shutil.rmtree(state_dir)
        print("  🗑  Cleared saved browser state")

    BROWSER_STATE_DIR.mkdir(parents=True, exist_ok=True)
    captured_token = None

    def on_request(request):
        nonlocal captured_token
        auth = request.headers.get("authorization", "")
        if (auth.startswith("Bearer ey")
                and "api.footytips.espn.com.au" in request.url):
            candidate = auth[7:]
            # Only accept tokens with ≥5 min validity — reject stale
            # tokens the browser sends before OneID refreshes.
            if _is_token_fresh(candidate, min_remaining=300):
                captured_token = candidate

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=state_dir,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )

        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.on("request", on_request)

        # ── Navigate ──────────────────────────────────────────────
        print("  🌐 Navigating to ESPN Footytips...")
        page.goto("https://footytips.espn.com.au/tipping/nrl",
                   wait_until="domcontentloaded", timeout=45000)

        # Wait for OneID + any silent refresh
        print("  ⏳ Waiting for OneID...")
        page.wait_for_timeout(10000)

        # If no valid token yet, OneID may still be refreshing —
        # wait longer with periodic checks (up to 20s more).
        if not captured_token:
            print("  ⏳ Waiting for token refresh...")
            for _ in range(4):
                page.wait_for_timeout(5000)
                if captured_token:
                    break

        if captured_token:
            print("  ✓ Token captured (silent refresh from saved session)")
            ctx.close()
            return captured_token

        # ── Check if login needed ─────────────────────────────────
        has_login_btn = page.locator('button:has-text("Log In")').count() > 0
        if not has_login_btn and captured_token:
            ctx.close()
            return captured_token

        if has_login_btn:
            print("  🔐 No saved session — need to log in")

            # Snapshot Gmail OTP thread BEFORE login triggers a new OTP
            print("  📋 Snapshotting Gmail OTP messages (pre-login)...")
            _gmail_snapshot = _snapshot_gmail_otp_ids()
            print(f"     {len(_gmail_snapshot[0])} existing OTP messages tracked")

            email = os.environ.get("FOOTYTIPS_EMAIL", "")
            password = os.environ.get("FOOTYTIPS_PASSWORD", "")
            if not email and sys.stdin.isatty():
                email = input("  Email: ").strip()
            if not password and sys.stdin.isatty():
                import getpass
                password = getpass.getpass("  Password: ").strip()

            if not email or not password:
                print("  ✗ Email and password are required")
                ctx.close()
                return None

            # Click login button
            print("  📧 Opening login form...")
            page.click('button:has-text("Log In")', timeout=5000)
            page.wait_for_timeout(3000)

            target = _find_oneid_frame(page) or page

            try:
                # ── Step 1: Email ──────────────────────────────────
                print("  ✏️  Filling email...")
                target.wait_for_selector(
                    "#InputIdentityFlowValue",
                    timeout=10000,
                )
                target.fill("#InputIdentityFlowValue", email)
                target.click("#BtnSubmit", timeout=5000)

                # ── Step 2: Password ───────────────────────────────
                print("  🔑 Waiting for password field...")
                target.wait_for_selector(
                    "#InputPassword",
                    state="visible",
                    timeout=15000,
                )
                page.wait_for_timeout(500)

                # Check for errors (e.g. account not found)
                err = target.locator('[class*="error"]:visible, [role="alert"]:visible')
                if err.count() > 0:
                    err_text = err.first.text_content().strip()
                    if err_text:
                        print(f"  ✗ OneID error: {err_text}")
                        ctx.close()
                        return None

                print("  ✏️  Filling password...")
                target.fill("#InputPassword", password)
                target.click("#BtnSubmit", timeout=5000)

                print("  ⏳ Logging in...")
                page.wait_for_timeout(12000)

                # ── Step 3: Handle OTP if required ─────────────────
                if _detect_otp_screen(target):
                    print("  📱 OTP verification required by Disney OneID")

                    # Try sources in order: CLI arg → env var → Gmail → interactive
                    if not otp_code:
                        otp_code = os.environ.get("FOOTYTIPS_OTP", "")
                    if not otp_code:
                        # Auto-fetch from Gmail via gmcli (using pre-login snapshot)
                        try:
                            otp_code = _poll_gmail_otp(
                                timeout=180,
                                pre_snapshot=_gmail_snapshot,
                            ) or ""
                        except Exception:
                            pass
                    if not otp_code and sys.stdin.isatty():
                        print("  📧 A 6-digit code was sent to your email.")
                        otp_code = input("  Enter OTP code: ").strip()

                    if not otp_code or len(otp_code) != 6:
                        print("  ✗ 6-digit OTP code required")
                        print("  💡 Re-run with: --otp <code>  or  FOOTYTIPS_OTP=<code>")
                        ctx.close()
                        return None

                    _enter_otp(target, page, otp_code)
                    print("  ⏳ Verifying OTP...")
                    page.wait_for_timeout(15000)

                    # Check for OTP errors
                    if _detect_otp_screen(target):
                        err = target.locator(
                            '[class*="error"]:visible, [role="alert"]:visible'
                        )
                        if err.count() > 0:
                            print(f"  ✗ OTP error: {err.first.text_content().strip()}")
                        else:
                            print("  ✗ OTP verification did not complete")
                        ctx.close()
                        return None

                    print("  ✓ OTP accepted")

                # Check for other login errors
                try:
                    err = target.locator('[class*="error"]:visible, [role="alert"]:visible')
                    if err.count() > 0:
                        err_text = err.first.text_content().strip()
                        if err_text:
                            print(f"  ✗ Login error: {err_text}")
                except Exception:
                    pass

            except Exception as e:
                print(f"  ✗ Login failed: {e}")
                ctx.close()
                return None

        # ── Try to trigger API calls if no token yet ──────────────
        if not captured_token:
            print("  🔄 Navigating to trigger API call...")
            page.goto("https://footytips.espn.com.au/tipping/nrl",
                       wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(10000)

        # ── Fallback: extract from localStorage / cookies ─────────
        if not captured_token:
            captured_token = _extract_token_fallbacks(page, ctx)

        ctx.close()
        return captured_token


def main():
    parser = argparse.ArgumentParser(description="ESPN Footytips Token Refresh")
    parser.add_argument("--check", action="store_true", help="Just check token status")
    parser.add_argument("--fresh", action="store_true", help="Force fresh login")
    parser.add_argument("--force", action="store_true", help="Refresh even if valid")
    parser.add_argument("--otp", type=str, default="", help="6-digit OTP code for email verification")
    args = parser.parse_args()

    is_valid, remaining = check_token_expiry()

    if args.check:
        if is_valid:
            print(f"✓ Token valid for {remaining // 3600}h {(remaining % 3600) // 60}m")
        else:
            print("✗ Token expired or missing")
        sys.exit(0 if is_valid else 1)

    if is_valid and not args.force and not args.fresh and remaining > 3600:
        h, m = remaining // 3600, (remaining % 3600) // 60
        print(f"✓ Token still valid ({h}h {m}m). Use --force to refresh.")
        return

    if not is_valid:
        print("⚠ Token expired or missing. Refreshing...")
    elif remaining <= 3600:
        print(f"⚠ Token expiring soon ({remaining // 60}m). Refreshing...")
    else:
        print("🔄 Force-refreshing token...")

    print()
    new_token = refresh_token(fresh=args.fresh, otp_code=args.otp)

    if not new_token:
        print("\n✗ Failed to obtain a new token")
        sys.exit(1)

    # Decode JWT metadata
    swid = ""
    try:
        payload = new_token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        jwt_data = json.loads(base64.urlsafe_b64decode(payload))
        swid = jwt_data.get("sub", "")
        remaining = jwt_data.get("exp", 0) - int(time.time())
        if remaining > 0:
            print(f"  🔑 New token valid for {remaining // 3600}h {(remaining % 3600) // 60}m")
        else:
            print(f"  ⚠ Token already expired ({abs(remaining) // 60}m ago) — refresh may have failed")
    except Exception:
        pass

    # Resolve member ID
    print("  🔍 Resolving member ID...")
    member_id = resolve_member_id(new_token) or ""
    if member_id:
        print(f"  👤 Member ID: {member_id}")

    save_creds(new_token, member_id=member_id, swid=swid)
    print(f"\n✅ Token saved to {CREDS_FILE}")


if __name__ == "__main__":
    main()
