#!/usr/bin/env python3
"""NRL-Predict Telegram Notification Module.

Sends tipping alerts to the same Telegram chat as Atlas.
Reads credentials from ~/.atlas-secrets.json (shared with Atlas).

Usage (CLI):
    python3 scripts/telegram_notify.py tips          # send tipping card
    python3 scripts/telegram_notify.py refresh       # send refresh summary
    python3 scripts/telegram_notify.py lineup        # send lineup change alert
    python3 scripts/telegram_notify.py error <step>  # send error alert
    python3 scripts/telegram_notify.py test          # connectivity test
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRETS_PATH = Path.home() / ".atlas-secrets.json"
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MSG_LEN = 4000


def _load_credentials() -> tuple[str, str]:
    """Return (bot_token, chat_id) from Atlas secrets file."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not (token and chat_id) and SECRETS_PATH.exists():
        with open(SECRETS_PATH) as f:
            secrets = json.load(f)
        token = token or secrets.get("telegram_bot_token", "")
        chat_id = chat_id or secrets.get("telegram_chat_id", "")

    if not token or not chat_id:
        raise ValueError(
            "Telegram credentials not found. "
            "Ensure ~/.atlas-secrets.json has telegram_bot_token and telegram_chat_id"
        )
    return token, chat_id


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_message(text: str, parse_mode: str = "HTML", silent: bool = False) -> bool:
    """Send a message to the configured Telegram chat."""
    try:
        token, chat_id = _load_credentials()
    except ValueError as e:
        logger.error("Telegram send failed: %s", e)
        return False

    if len(text) > MAX_MSG_LEN:
        text = text[:MAX_MSG_LEN - 20] + "\n\n… (truncated)"

    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_notification": silent,
        "disable_web_page_preview": True,
    }).encode("utf-8")

    url = TELEGRAM_API.format(token=token)
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
            if body.get("ok"):
                logger.info("Telegram message sent")
                return True
            logger.warning("Telegram API returned ok=false: %s", body)
            return False
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        logger.error("Telegram HTTP %d: %s", e.code, body)
        return False
    except Exception as e:
        logger.error("Telegram send error: %s", e)
        return False


# ---------------------------------------------------------------------------
# NRL-specific message builders
# ---------------------------------------------------------------------------

def send_tipping_card(round_num: int | None = None, year: int = 2026) -> bool:
    """Send the tipping card for the current round."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Find the latest predictions CSV
    pred_dir = PROJECT_ROOT / "outputs" / "predictions"
    if round_num:
        csv_path = pred_dir / f"round_{round_num}_{year}.csv"
    else:
        csvs = sorted(pred_dir.glob(f"round_*_{year}.csv"))
        if not csvs:
            return send_message(
                f"🏉 <b>NRL Tipping — No predictions found</b>\n"
                f"<i>{now}</i>\n\nRun: python predict_round.py --auto"
            )
        csv_path = csvs[-1]
        # Extract round number from filename
        round_num = int(csv_path.stem.split("_")[1])

    if not csv_path.exists():
        return send_message(
            f"🏉 <b>NRL Tipping — Round {round_num}</b>\n"
            f"<i>{now}</i>\n\n⚠️ No predictions file found."
        )

    import pandas as pd
    df = pd.read_csv(csv_path)

    # Build tip summary
    lines = [
        f"🏉 <b>NRL Tipping Card — Round {round_num}, {year}</b>",
        f"<i>{now}</i>",
        "",
    ]

    # Categorise games
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        tip = row["tip"]
        prob = row["home_win_prob"]
        odds_prob = row.get("odds_home_prob", 0.5)

        fav_prob = max(odds_prob, 1 - odds_prob)
        if fav_prob >= 0.65:
            cat = "🔒"  # LOCK
        elif fav_prob >= 0.55:
            cat = "👉"  # LEAN
        else:
            cat = "🎲"  # TOSS-UP

        # Check if model flipped the favourite
        odds_fav = home if odds_prob > 0.5 else away
        flip = " ↩️" if tip != odds_fav else ""

        # Short team names
        home_short = home.split()[-1]
        away_short = away.split()[-1]

        conf_pct = abs(prob - 0.5) * 200
        lines.append(
            f"{cat} <b>{_esc(home_short)}</b> v <b>{_esc(away_short)}</b>"
            f" → <b>{_esc(tip.split()[-1])}</b>"
            f" ({conf_pct:.0f}%){flip}"
        )

    # Summary
    n_locks = sum(1 for _, r in df.iterrows()
                  if max(r.get("odds_home_prob", 0.5), 1 - r.get("odds_home_prob", 0.5)) >= 0.65)
    n_flips = sum(1 for _, r in df.iterrows()
                  if r["tip"] != (r["home_team"] if r.get("odds_home_prob", 0.5) > 0.5 else r["away_team"]))

    lines.append("")
    lines.append(f"🔒 {n_locks} locks  |  ↩️ {n_flips} model flips")

    return send_message("\n".join(lines))


def send_refresh_summary(round_num: int | None = None, year: int = 2026) -> bool:
    """Send a summary of the weekly data refresh."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    import pandas as pd

    lines = [
        f"🔄 <b>NRL Data Refresh</b>",
        f"<i>{now}</i>",
        "",
    ]

    # Check data files
    processed = PROJECT_ROOT / "data" / "processed"
    for name, path in [
        ("Matches", processed / "matches.parquet"),
        ("Player Appearances", processed / "player_appearances.parquet"),
        ("Player Impact", processed / "player_impact.parquet"),
    ]:
        if path.exists():
            df = pd.read_parquet(path)
            lines.append(f"  ✅ {name}: {len(df):,} rows")
        else:
            lines.append(f"  ❌ {name}: missing")

    if round_num:
        lines.append(f"\n  Round refreshed: {round_num}")

    lines.append(f"\n  Next: <code>python predict_round.py --auto</code>")

    return send_message("\n".join(lines))


def send_lineup_alert(round_num: int, year: int = 2026) -> bool:
    """Send lineup change alert with impact scores."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # This is called after --check-lineups has run, so we read
    # from the cached team list data
    cache_path = PROJECT_ROOT / "data" / "teamlists" / f"round_{round_num}_{year}.json"
    if not cache_path.exists():
        return send_message(
            f"🔄 <b>NRL Lineup Check — Round {round_num}</b>\n"
            f"<i>{now}</i>\n\nNo team list data available yet."
        )

    return send_message(
        f"📋 <b>NRL Team Lists — Round {round_num}, {year}</b>\n"
        f"<i>{now}</i>\n\n"
        f"Team lists fetched from NRL.com.\n"
        f"Run <code>python predict_round.py --auto --check-lineups</code> "
        f"for impact-adjusted tipping card."
    )


def send_submit_confirmation(
    round_num: int, n_tips: int, year: int = 2026
) -> bool:
    """Send confirmation that tips were submitted to ESPN Footytips."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Load predictions for the card
    pred_dir = PROJECT_ROOT / "outputs" / "predictions"
    csv_path = pred_dir / f"round_{round_num}_{year}.csv"

    lines = [
        f"🏉 <b>Tips Submitted — Round {round_num}, {year}</b>",
        f"<i>{now}</i>",
        "",
    ]

    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            tip = row["tip"]
            prob = row["home_win_prob"]
            odds_prob = row.get("odds_home_prob", 0.5)

            fav_prob = max(odds_prob, 1 - odds_prob)
            if fav_prob >= 0.65:
                cat = "🔒"
            elif fav_prob >= 0.55:
                cat = "👉"
            else:
                cat = "🎲"

            odds_fav = home if odds_prob > 0.5 else away
            flip = " ↩️" if tip != odds_fav else ""
            home_short = home.split()[-1]
            away_short = away.split()[-1]
            conf_pct = abs(prob - 0.5) * 200
            lines.append(
                f"{cat} <b>{_esc(home_short)}</b> v <b>{_esc(away_short)}</b>"
                f" → <b>{_esc(tip.split()[-1])}</b>"
                f" ({conf_pct:.0f}%){flip}"
            )

    lines.append("")
    lines.append(f"✅ <b>{n_tips} tips submitted</b> to ESPN Footytips")

    return send_message("\n".join(lines))


def send_error(step: str, detail: str = "", logfile: str | None = None) -> bool:
    """Send an error alert for a failed NRL cron step."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    log_tail = ""
    if logfile and Path(logfile).exists():
        lines = Path(logfile).read_text(errors="replace").splitlines()
        tail = lines[-10:] if len(lines) > 10 else lines
        log_tail = "\n\n<b>Log tail:</b>\n<pre>" + _esc("\n".join(tail)) + "</pre>"

    msg = (
        f"🚨 <b>NRL CRON FAILED</b>\n"
        f"<i>{now}</i>\n\n"
        f"<b>Step:</b> {_esc(step)}\n"
    )
    if detail:
        msg += f"<b>Error:</b>\n<pre>{_esc(detail[:1500])}</pre>"
    msg += log_tail

    return send_message(msg)


def send_test() -> bool:
    """Send a connectivity test message."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return send_message(
        f"🏉 <b>NRL-Predict Telegram Connected</b>\n"
        f"<i>{now}</i>\n\n"
        f"Alerts are active. You'll receive:\n"
        f"  🔄 Monday refresh confirmations\n"
        f"  🏉 Tuesday tipping cards\n"
        f"  📋 Game-day lineup change alerts\n"
        f"  🚨 Error alerts for failed crons"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    cmd = sys.argv[1]

    if cmd == "tips":
        round_num = int(sys.argv[2]) if len(sys.argv) > 2 else None
        ok = send_tipping_card(round_num)
    elif cmd == "submit":
        round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        n_tips = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        ok = send_submit_confirmation(round_num, n_tips)
    elif cmd == "refresh":
        round_num = int(sys.argv[2]) if len(sys.argv) > 2 else None
        ok = send_refresh_summary(round_num)
    elif cmd == "lineup":
        round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        ok = send_lineup_alert(round_num)
    elif cmd == "error":
        step = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        logfile = sys.argv[3] if len(sys.argv) > 3 else None
        ok = send_error(step, logfile=logfile)
    elif cmd == "test":
        ok = send_test()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
