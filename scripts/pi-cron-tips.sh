#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# NRL Tuesday Tipping — Pi Agent Wrapper
# ──────────────────────────────────────────────────────────────────
#
# Invokes a pi agent to run the full Tuesday tipping pipeline:
#   1. Generate predictions (model retrain + fresh odds)
#   2. Sanity-check blend consistency
#   3. Submit tips to ESPN Footytips
#   4. Verify submission via API
#   5. Send Telegram confirmation
#
# The agent can autonomously diagnose and fix errors (e.g., token
# refresh, blend bugs, submission failures) instead of just failing.
#
# Crontab:
#   TZ=Australia/Brisbane
#   0 17 * * 2  /root/NRL-Predict/scripts/pi-cron-tips.sh
#
# Manual run:
#   /root/NRL-Predict/scripts/pi-cron-tips.sh
#
# ──────────────────────────────────────────────────────────────────

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
SKILL_DIR="$PROJECT_ROOT/.pi/skills/nrl-tuesday-tips"
NOTIFY="python3 $SCRIPT_DIR/telegram_notify.py"

mkdir -p "$LOG_DIR"

export TZ="Australia/Brisbane"
export HOME="${HOME:-/root}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOGFILE="$LOG_DIR/pi-cron-tips-${TIMESTAMP}.log"

# Symlink latest log for easy reference
ln -sf "pi-cron-tips-${TIMESTAMP}.log" "$LOG_DIR/pi-cron-tips-latest.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
notify() { $NOTIFY "$@" 2>>"$LOG_DIR/telegram.log" || true; }

echo "$(ts) ── Pi agent tipping starting ──" >> "$LOGFILE"

cd "$PROJECT_ROOT"

# ── Invoke pi agent ───────────────────────────────────────────────
# --print:      non-interactive, prints output and exits
# --skill:      loads the nrl-tuesday-tips skill for pipeline instructions
# --no-session: ephemeral, no session file clutter
#
# The agent reads the skill, follows the pipeline steps, and handles
# any errors it encounters. Output is captured to the log file.
# ──────────────────────────────────────────────────────────────────

PROMPT="Run the NRL Tuesday tipping pipeline now. Follow all 6 steps in the nrl-tuesday-tips skill. Generate predictions, sanity-check, submit to ESPN Footytips, verify, and send Telegram confirmation. Fix any errors you encounter — it is critical that tips are submitted correctly. Write a brief summary of results to $LOG_DIR/pi-cron-tips-${TIMESTAMP}.md when done."

pi --print \
   --skill "$SKILL_DIR" \
   --no-session \
   "$PROMPT" \
   >> "$LOGFILE" 2>&1

EXIT_CODE=$?

echo "$(ts) ── Pi agent finished (exit=$EXIT_CODE) ──" >> "$LOGFILE"

# ── Fallback: if the agent itself crashes, alert via Telegram ─────
if [ $EXIT_CODE -ne 0 ]; then
    echo "$(ts) ── Pi agent FAILED — sending error alert ──" >> "$LOGFILE"
    notify error "tuesday-tips-agent" "$LOGFILE"
fi

exit $EXIT_CODE
