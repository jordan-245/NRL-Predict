#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# NRL-Predict Cron Wrapper
# ──────────────────────────────────────────────────────────────────
#
# Called by crontab entries. Handles:
#   1. Monday evening    → data refresh (results + lineups + impact)
#   2. Tuesday afternoon → generate tipping card + send to Telegram
#   3. Thu-Sun game days → check lineups + send adjustments
#
# Usage:
#   scripts/nrl-cron.sh refresh      # Monday: post-round data refresh
#   scripts/nrl-cron.sh tips         # Tuesday: generate + send tips
#   scripts/nrl-cron.sh lineups      # Game day: check lineup changes
#
# All output is logged to logs/nrl-cron-<mode>.log
# Errors trigger a Telegram alert.
# ──────────────────────────────────────────────────────────────────

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_ROOT/.venv/bin/python3"
NOTIFY="$VENV $SCRIPT_DIR/telegram_notify.py"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

MODE="${1:-help}"
LOG="$LOG_DIR/nrl-cron-${MODE}.log"

# Timestamp helper
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Telegram notify (swallow errors so they don't break the pipeline)
notify() { $NOTIFY "$@" 2>>"$LOG_DIR/telegram.log" || true; }

cd "$PROJECT_ROOT"

case "$MODE" in

# ── REFRESH (Monday 8pm AEST) ──────────────────────────────────
refresh)
    echo "$(ts) ── NRL refresh starting ──" >> "$LOG"

    if $VENV refresh_week.py --record-tips >> "$LOG" 2>&1; then
        echo "$(ts) ── NRL refresh complete ──" >> "$LOG"
        notify refresh
    else
        echo "$(ts) ── NRL refresh FAILED ──" >> "$LOG"
        notify error "refresh" "$LOG"
    fi

    ;;

# ── TIPS (Tuesday 5pm AEST) ────────────────────────────────────
tips)
    echo "$(ts) ── NRL tipping starting ──" >> "$LOG"

    if $VENV predict_round.py --auto --retrain >> "$LOG" 2>&1; then
        echo "$(ts) ── NRL tipping complete ──" >> "$LOG"
        notify tips

        # ── Auto-submit tips to ESPN Footytips ──
        echo "$(ts) ── ESPN Footytips auto-submit starting ──" >> "$LOG"

        # Refresh Disney OneID token (uses saved browser session)
        if $VENV scripts/footytips_auth.py >> "$LOG" 2>&1; then
            echo "$(ts) ── Token refreshed ──" >> "$LOG"
        else
            echo "$(ts) ── Token refresh failed, trying with existing token ──" >> "$LOG"
        fi

        # Detect current round from the latest prediction file
        ROUND=$( ls -1 outputs/predictions/round_*_*.csv 2>/dev/null \
                 | sort -t_ -k2 -n | tail -1 \
                 | sed 's/.*round_\([0-9]*\)_.*/\1/' )

        if [ -n "$ROUND" ]; then
            if $VENV scripts/footytips_submit.py --round "$ROUND" --joker >> "$LOG" 2>&1; then
                echo "$(ts) ── ESPN tips submitted for round $ROUND ──" >> "$LOG"
                notify submit "$ROUND"
            else
                echo "$(ts) ── ESPN tip submit FAILED ──" >> "$LOG"
                notify error "footytips-submit" "$LOG"
            fi
        else
            echo "$(ts) ── No prediction file found, skipping submit ──" >> "$LOG"
        fi
    else
        echo "$(ts) ── NRL tipping FAILED ──" >> "$LOG"
        notify error "tips" "$LOG"
    fi

    ;;

# ── LINEUPS (Thu-Sun 3pm AEST) ─────────────────────────────────
lineups)
    echo "$(ts) ── NRL lineup check starting ──" >> "$LOG"

    if $VENV predict_round.py --auto --check-lineups >> "$LOG" 2>&1; then
        echo "$(ts) ── NRL lineup check complete ──" >> "$LOG"
        notify tips
    else
        echo "$(ts) ── NRL lineup check FAILED ──" >> "$LOG"
        notify error "lineups" "$LOG"
    fi

    ;;

# ── HELP ────────────────────────────────────────────────────────
help|*)
    echo "Usage: $0 {refresh|tips|lineups}"
    echo ""
    echo "  refresh   Monday: post-round data refresh"
    echo "  tips      Tuesday: generate + send tipping card"
    echo "  lineups   Game day: check lineup changes + rescore"
    exit 0
    ;;

esac

exit 0
