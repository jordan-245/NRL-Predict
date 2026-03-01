#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# NRL-Predict Cron Wrapper
# ──────────────────────────────────────────────────────────────────
#
# Modes:
#   refresh   Monday 8pm AEST:  scrape last round, rebuild data
#   tips      Tuesday 5pm AEST: predict + submit ALL tips to ESPN
#   pregame   Thu-Sun (every 30 min): lineup check → re-tip on swings
#
# Auth: Disney OneID tokens last ~24h. Token is only refreshed when
# it has <2h remaining, to avoid unnecessary Disney API hits.
#
# Crontab:
#   TZ=Australia/Brisbane
#   0 20 * * 1      scripts/nrl-cron.sh refresh
#   0 17 * * 2      scripts/nrl-cron.sh tips
#   */30 13-20 * * 4-7  scripts/nrl-cron.sh pregame
# ──────────────────────────────────────────────────────────────────

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PY="python3"
NOTIFY="$PY $SCRIPT_DIR/telegram_notify.py"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

MODE="${1:-help}"
LOG="$LOG_DIR/nrl-cron-${MODE}.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
notify() { $NOTIFY "$@" 2>>"$LOG_DIR/telegram.log" || true; }

cd "$PROJECT_ROOT"

# ── Ensure token is valid (only refresh if <2h left) ──────────────
ensure_token() {
    echo "$(ts) Checking ESPN token..." >> "$LOG"

    # --check exits 0 if valid, 1 if not
    if $PY scripts/footytips_auth.py --check >> "$LOG" 2>&1; then
        # Token exists and is valid — check if expiring soon (<2h)
        REMAINING=$($PY -c "
from scripts.footytips_auth import check_token_expiry
valid, secs = check_token_expiry()
print(secs)
" 2>/dev/null || echo "0")

        if [ "$REMAINING" -lt 7200 ] 2>/dev/null; then
            echo "$(ts) Token expiring soon (${REMAINING}s) — refreshing..." >> "$LOG"
            if ! $PY scripts/footytips_auth.py >> "$LOG" 2>&1; then
                echo "$(ts) ⚠ Token refresh failed — using existing token" >> "$LOG"
            fi
        else
            echo "$(ts) Token valid (${REMAINING}s remaining) — no refresh needed" >> "$LOG"
        fi
    else
        echo "$(ts) Token expired/missing — refreshing..." >> "$LOG"
        if ! $PY scripts/footytips_auth.py >> "$LOG" 2>&1; then
            echo "$(ts) ✗ Token refresh FAILED" >> "$LOG"
            notify error "token-refresh" "$LOG"
            return 1
        fi
    fi
    return 0
}

# ── Detect current round from latest prediction file ──────────────
detect_round() {
    ls -1 outputs/predictions/round_*_*.csv 2>/dev/null \
        | sort -t_ -k2 -n | tail -1 \
        | sed 's/.*round_\([0-9]*\)_.*/\1/'
}

case "$MODE" in

# ── REFRESH (Monday 8pm AEST) ──────────────────────────────────
# 1. Scrape last round, rebuild data, record results
# 2. Plan the week: fetch kickoff times, set pregame cron schedule
refresh)
    echo "$(ts) ── NRL refresh starting ──" >> "$LOG"

    if ! $PY refresh_week.py --record-tips >> "$LOG" 2>&1; then
        echo "$(ts) ── NRL refresh FAILED ──" >> "$LOG"
        notify error "refresh" "$LOG"
        exit 1
    fi
    echo "$(ts) ── NRL refresh complete ──" >> "$LOG"

    # Plan the week: fetch kickoff times, update pregame cron
    echo "$(ts) ── Planning week schedule ──" >> "$LOG"
    if $PY scripts/plan_week.py >> "$LOG" 2>&1; then
        echo "$(ts) ── Week schedule planned ──" >> "$LOG"

        # Check for tip deadline warnings
        TIP_WARNING=$($PY -c "
import json
s = json.load(open('config/week_schedule.json'))
w = s.get('tip_warning')
if w: print(w)
" 2>/dev/null)

        if [ -n "$TIP_WARNING" ]; then
            echo "$(ts) ⚠ $TIP_WARNING" >> "$LOG"
        fi
    else
        echo "$(ts) ⚠ Week schedule planning failed (non-critical)" >> "$LOG"
    fi

    notify refresh
    ;;

# ── TIPS (Tuesday 5pm AEST) ────────────────────────────────────
# 1. Retrain model with fresh data + live odds
# 2. Refresh ESPN token (only if needed)
# 3. Submit ALL tips
# 4. Verify submission count
# 5. Telegram with full tipping card
tips)
    echo "$(ts) ── NRL Tuesday tipping starting ──" >> "$LOG"

    # Step 1: Generate predictions
    if ! $PY predict_round.py --auto --retrain >> "$LOG" 2>&1; then
        echo "$(ts) ── Prediction FAILED ──" >> "$LOG"
        notify error "predict" "$LOG"
        exit 1
    fi
    echo "$(ts) Predictions generated" >> "$LOG"

    # Step 2: Ensure token is valid
    if ! ensure_token; then
        echo "$(ts) ── Cannot submit tips (no token) ──" >> "$LOG"
        notify error "tips-no-token" "$LOG"
        exit 1
    fi

    # Step 3: Submit tips
    ROUND=$(detect_round)
    if [ -z "$ROUND" ]; then
        echo "$(ts) ── No prediction file found ──" >> "$LOG"
        notify error "tips-no-predictions" "$LOG"
        exit 1
    fi

    echo "$(ts) Submitting tips for Round $ROUND..." >> "$LOG"
    SUBMIT_OUTPUT=$($PY scripts/footytips_submit.py --round "$ROUND" 2>&1)
    SUBMIT_RC=$?
    echo "$SUBMIT_OUTPUT" >> "$LOG"

    if [ $SUBMIT_RC -eq 0 ]; then
        # Extract tip count from output
        N_TIPS=$(echo "$SUBMIT_OUTPUT" | grep -oP 'Submitting \K\d+' | head -1)
        N_TIPS=${N_TIPS:-0}

        echo "$(ts) ── Tips submitted: $N_TIPS tips ──" >> "$LOG"

        # Step 4: Verify tips were actually saved
        VERIFY_COUNT=$($PY -c "
import sys; sys.path.insert(0, '.')
from scripts.footytips_submit import load_creds, get_auth_header, get_round_data
creds = load_creds(); headers = get_auth_header(creds)
data = get_round_data($ROUND, headers)
print(len(data.get('tips', [])))
" 2>/dev/null || echo "0")

        if [ "$VERIFY_COUNT" -gt 0 ] 2>/dev/null; then
            echo "$(ts) ✓ Verified: $VERIFY_COUNT tips on file" >> "$LOG"
            notify submit "$ROUND" "$N_TIPS"
        else
            echo "$(ts) ⚠ Verification: only $VERIFY_COUNT tips on file!" >> "$LOG"
            notify error "tips-verify-failed" "$LOG"
        fi
    else
        echo "$(ts) ── Tip submission FAILED ──" >> "$LOG"
        notify error "tips-submit" "$LOG"
        exit 1
    fi
    ;;

# ── PREGAME (Thu-Sun every 30 min, 1pm-8pm AEST) ──────────────
# Checks games kicking off in 30-90 min window.
# Only re-submits if lineup changes cause a tip FLIP.
pregame)
    echo "$(ts) ── Pre-game check starting ──" >> "$LOG"

    # Ensure token (only refresh if needed)
    if ! ensure_token; then
        echo "$(ts) ── Pre-game check skipped (no token) ──" >> "$LOG"
        notify error "pregame-no-token" "$LOG"
        exit 1
    fi

    if $PY scripts/pregame_check.py >> "$LOG" 2>&1; then
        echo "$(ts) ── Pre-game check complete ──" >> "$LOG"
    else
        echo "$(ts) ── Pre-game check FAILED ──" >> "$LOG"
        notify error "pregame" "$LOG"
    fi
    ;;

# ── HELP ────────────────────────────────────────────────────────
help|*)
    echo "Usage: $0 {refresh|tips|pregame}"
    echo ""
    echo "  refresh   Monday 8pm:  post-round data refresh"
    echo "  tips      Tuesday 5pm: predict + submit all tips"
    echo "  pregame   Thu-Sun:     lineup check → re-tip on swings"
    exit 0
    ;;

esac

exit 0
