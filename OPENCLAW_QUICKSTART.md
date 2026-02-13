# NRL-Predict + OpenClaw + Telegram Alerts (Quickstart)

Goal: run this repo's tipping workflow in OpenClaw, and get Telegram alerts 60 minutes before every match with the tip + reasoning + margin.

This doc is written so an automation agent (or you) can implement it quickly and repeatably.

---

## What This Repo Does (High Level)

Data in -> features -> model -> tips -> alerts -> tracking

1. Historical data build (one-off / occasional)
   - `run_scrape.py` scrapes match results + ladders from RugbyLeagueProject and joins historical odds from the AusSportsBetting Excel file.
   - Outputs Parquet files into `data/processed/` (ignored by git).

2. Weekly predictions (Tuesday)
   - `tipping_advisor.py --auto` calls the prediction pipeline (`predict_round.py`) to:
     - fetch next round fixtures + odds from The Odds API
     - build features (V3 + V4 feature engineering)
     - train (or re-score from cache) an ensemble
     - write a predictions CSV to `outputs/predictions/round_<N>_<YEAR>.csv`
     - print a tipping card

3. Tipping logic (simple + disciplined)
   - LOCK (fav >= 65%): always tip the favourite
   - LEAN (55-65%): default favourite; model can flip if it strongly disagrees
   - TOSS-UP (<55%): use the model
   - MARGIN: use spread if available, else odds-implied margin

4. Results tracking (weekly)
   - `tipping_tracker.py` logs your tips + outcomes and keeps a season dashboard in `outputs/tipping_log_<YEAR>.csv`.

---

## Why This Strategy (2025 Benchmarks)

This repo's tipping advisor is designed around what actually mattered in 2025:

- Favourite accuracy was 62.4% (133/213), not ~68%.
- You got 136, beating the favourite baseline by +3 tips.
- Close games were effectively coin flips (favourite < 60% won 51.7%).
- There were 58 close games in 2025; you only need a small edge on these to win.
- Margin tiebreaker matters: using spread/odds-implied margin is a repeatable edge over gut feel.

The system focuses modelling effort where it can move your season outcome (LEANS + TOSS-UPS) and removes decision fatigue everywhere else (LOCKS).

---

## Key Files / Entrypoints

Run these from the repo root:

- `run_scrape.py`
  - Builds `data/processed/matches.parquet`, `data/processed/ladders.parquet`, `data/processed/odds.parquet`.
- `predict_round.py`
  - Core modelling pipeline (features + training + caching + predictions CSV).
- `tipping_advisor.py`
  - Produces the tipping card (manual mode or auto mode).
- `tipping_tracker.py`
  - Records results and shows dashboards.
- `scraping/odds_api.py`
  - The Odds API integration (fixtures, odds, scores).

---

## Repo Layout (So You Know Where Things Live)

- `config/`
  - Project settings, team name mappings, Elo params.
- `data/raw/`
  - Scraped/cached source files (mostly ignored by git).
  - Historical odds Excel expected at `data/raw/odds/nrl_odds.xlsx` if you re-run `run_scrape.py`.
- `data/processed/`
  - Parquet files used by training/inference (ignored by git, but required at runtime).
- `outputs/model_cache/`
  - Per-round model artifacts cached as `.joblib` for fast re-scoring with fresh odds.
- `outputs/predictions/`
  - Per-round predictions CSVs (what alerts should read).

---

## Prereqs

### Python (this repo)

- Python >= 3.10 (see `pyproject.toml`)
- Install deps:

```powershell
cd C:\Users\jorda\Development\NRL-Predict
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

### The Odds API key

Create `.env` in the repo root:

```ini
ODDS_API_KEY=YOUR_KEY
```

Notes:
- `tipping_advisor.py --auto` uses The Odds API.
- Odds API calls have a credit cost for odds/scores; fixtures are free (see `scraping/odds_api.py`).

Odds API credit usage (from `scraping/odds_api.py`):
- Fixtures/events: free
- Odds (h2h/spreads): costs credits
- Scores: costs credits

### OpenClaw (runtime + scheduler + Telegram delivery)

OpenClaw runs a long-lived Gateway process, supports cron jobs, and can deliver messages to Telegram.

Start here:
- https://docs.openclaw.ai/getting-started
- https://docs.openclaw.ai/configuration
- https://docs.openclaw.ai/channels/telegram
- https://docs.openclaw.ai/automation/cron-jobs
- https://docs.openclaw.ai/automation/heartbeat

OpenClaw config file locations:
- Linux/macOS: `~/.openclaw/openclaw.json`
- Windows: `%USERPROFILE%\.openclaw\openclaw.json`

---

## Local Smoke Test (No OpenClaw Yet)

Run the advisor in auto mode:

```powershell
python tipping_advisor.py --auto --year 2026
```

Expected outcomes:
- Prints a tipping card with LOCK/LEAN/TOSS-UP buckets.
- Writes `outputs/predictions/round_<N>_2026.csv`.

If you re-run the same round, it should use cached models and just re-score with fresh odds (fast path) unless you force retraining.

### Common commands

```powershell
# Auto: fixtures + odds from The Odds API, round auto-detected
python tipping_advisor.py --auto --year 2026

# Auto: force a specific round
python tipping_advisor.py --auto --round 5 --year 2026

# Just the prediction pipeline (prints the same tipping card format)
python predict_round.py --auto --year 2026

# Force retrain (ignore model cache)
python predict_round.py --auto --retrain --year 2026

# Retune Elo (slow)
python predict_round.py --auto --retune-elo --year 2026

# Track results
python tipping_tracker.py --record 1 --auto --year 2026

# Dashboard
python tipping_tracker.py --year 2026
```

Model cache notes:
- Saved under `outputs/model_cache/` as `.joblib`.
- Used to re-score quickly with fresh odds without retraining.

---

## Timezones (Do Not Skip This)

- The Odds API returns `commence_time` in UTC.
- This repo stores upcoming match `date` as a UTC moment, but when features are built the timezone is dropped (it becomes a timezone-naive timestamp representing the UTC time).

For alerts:
- Treat `date` in `outputs/predictions/round_<N>_<YEAR>.csv` as UTC.
- Prefer scheduling one-shot cron jobs using `--at "...Z"` timestamps so you are not dependent on the Gateway machine's local timezone.

---

## Weekly Workflow + Schedules (Human Process)

All times below assume Australia/Sydney (NRL context). Convert if you're elsewhere.

### Pre-season / data refresh (occasional)

- Run `python run_scrape.py` to rebuild `data/processed/*.parquet` (matches, ladders, odds).
- Recommended cadence:
  - Pre-season: once (late Feb / early March)
  - In-season: weekly (after the round finishes) if you want ladder/form features to reflect the current season

### Tuesday (team lists + odds day)

- 16:00: Team lists published (manual workflow trigger).
- 16:05-18:00:
  - Run: `python tipping_advisor.py --auto`
  - Result: predictions saved + tipping card.
- After tips (optional but recommended):
  - Schedule alert jobs for each match (OpenClaw cron, see below).

### Match day (each match)

- T - 60 minutes:
  - Telegram alert with:
    - tip, category, confidence, reasoning, suggested margin
    - reminder: check final team list for late spine outs (HB/FB/Hooker)

### After the round finishes

- Run:
  - `python tipping_tracker.py --record <ROUND> --auto`
  - This logs outcomes and shows season progress.

---

## OpenClaw Setup (Minimum Viable)

### 1) Install + run Gateway

Follow the OpenClaw onboarding wizard and ensure the Gateway is always running.

### 2) Configure Telegram channel

Minimal config example (edit your OpenClaw config; JSON5 is allowed):

```json5
{
  channels: {
    telegram: {
      enabled: true,
      botToken: "123:abc",
      dmPolicy: "pairing",
    },
  },
}
```

Finding your chat/user id:
- DM your bot, then watch `openclaw logs --follow` for `from.id`.
- Or use Telegram Bot API `getUpdates` and read `message.from.id`.

### 3) Point the agent workspace at this repo

Set `agents.defaults.workspace` to this repo path so the agent can run the scripts here.

Examples:
- Windows native: `C:\\Users\\jorda\\Development\\NRL-Predict`
- WSL2: `/mnt/c/Users/jorda/Development/NRL-Predict`

---

## OpenClaw Automation Plan (What To Implement)

Use cron for "exactly 60 minutes before kickoff" alerts.

Important: `tipping_advisor.py --auto` consumes Odds API credits. Do it once per round (Tuesday), then reuse the saved `outputs/predictions/round_<N>_<YEAR>.csv` for all match alerts.

### Strategy A (Recommended): Create one-shot "AT" cron jobs per match

1. Tuesday job runs `tipping_advisor.py --auto` to generate `outputs/predictions/round_<N>_<YEAR>.csv`.
2. The same Tuesday job reads that CSV and creates one-shot cron jobs for:
   - kickoff_time_utc - 60 minutes
3. Each match cron job runs in an isolated session and announces to Telegram.

What to implement in this repo:

1. `scripts/render_match_alert.py`
   - Inputs: `--predictions-csv`, `--home`, `--away` (or a match id)
   - Output: plain text message (Telegram-friendly; no tables)

2. `scripts/schedule_openclaw_alerts.py`
   - Inputs: `--year`, `--round` (optional auto), `--to <telegram_chat_id>`
   - Steps:
     - run `python tipping_advisor.py --auto --year <YEAR>` (once)
     - load the generated predictions CSV
     - for each match row:
       - compute `alert_at = kickoff_utc - 1 hour`
       - call:
         - `openclaw cron add --name ... --at <alert_at_iso_z> --session isolated --message "<instruction>" --announce --channel telegram --to "<chat_id>"`
   - Must be idempotent: if jobs already exist for that round, update/replace them.

Optional but useful:

3. `scripts/refresh_processed_data.py`
   - Goal: keep `data/processed/*.parquet` current during the season (ladders/form features).
   - Implementation: run `python run_scrape.py` (or a faster "current season only" scrape, if you add it).

4. `scripts/record_last_round.py`
   - Goal: automatically log the most recently completed round with `tipping_tracker.py`.
   - Implementation idea:
     - detect last completed NRL round (from `data/processed/matches.parquet` or Odds API scores)
     - run `python tipping_tracker.py --record <ROUND> --auto`

Cron notes:
- `--at` takes an ISO timestamp. Use an explicit `Z` timezone (UTC) to avoid ambiguity.
- Telegram `--to` can be a chat id like `-100...` (groups/supergroups) or `-100...:topic:<id>` (forum topic).

### Strategy B (Simpler, less precise): poll every 10 minutes

Instead of per-match jobs, use a repeating cron job every `10m`:
- load the latest predictions CSV
- if any match is within the next 60 minutes and not alerted yet, send it
- write state to `outputs/alert_state.json` to avoid duplicates

This is easier to implement but timing can drift and you must handle dedupe carefully.

---

## OpenClaw Cron Schedules (Concrete)

Suggested job roster (all in Australia/Sydney unless noted):

- Weekly data refresh (optional): Mon 09:00 -> update `data/processed/*.parquet`
- Weekly tips + alert scheduling: Tue 16:10 -> create predictions + create per-match alert jobs
- Per match alerts: kickoff - 60m (scheduled as one-shot `--at ...Z` UTC)
- Weekly results logging (optional): Mon 10:00 -> record last completed round

### 1) Weekly "Refresh + schedule alerts" (Tuesday)

Create a recurring cron job at Tuesday ~16:10 Australia/Sydney:

- Kind: `cron`
- Expr: `10 16 * * 2` (Tue 16:10)
- TZ: `Australia/Sydney`
- Session: `isolated`
- Delivery: `announce` to Telegram (optional summary)

CLI shape (template):

```bash
openclaw cron add \
  --name "NRL: refresh tips + schedule alerts" \
  --cron "10 16 * * 2" \
  --tz "Australia/Sydney" \
  --session isolated \
  --message "In repo workspace, run the weekly NRL tipping refresh and schedule all match alerts for the upcoming round." \
  --announce \
  --channel telegram \
  --to "<YOUR_CHAT_ID>"
```

### (Optional) Weekly data refresh (Monday)

If you want ladders/form features to stay current during the season, schedule a weekly refresh.

```bash
openclaw cron add \
  --name "NRL: refresh processed data" \
  --cron "0 9 * * 1" \
  --tz "Australia/Sydney" \
  --session isolated \
  --message "In repo workspace, run scripts/refresh_processed_data.py (or run_scrape.py) to update data/processed/*.parquet." \
  --announce \
  --channel telegram \
  --to "<YOUR_CHAT_ID>"
```

### (Optional) Weekly results logging (Monday)

If you implement `scripts/record_last_round.py`, you can schedule it weekly after the round.

```bash
openclaw cron add \
  --name "NRL: record last round" \
  --cron "0 10 * * 1" \
  --tz "Australia/Sydney" \
  --session isolated \
  --message "In repo workspace, run scripts/record_last_round.py to log the most recently completed round." \
  --announce \
  --channel telegram \
  --to "<YOUR_CHAT_ID>"
```

### 2) Match alerts (one-shot per match)

For each match row in predictions:

- Kind: `at`
- At: kickoff_utc_minus_1h as ISO with `Z`
- Session: `isolated`
- Delivery: `announce` to Telegram

CLI shape (template):

```bash
openclaw cron add \
  --name "NRL: <HOME> v <AWAY> (T-60m)" \
  --at "2026-03-06T08:50:00Z" \
  --session isolated \
  --message "Run scripts/render_match_alert.py for <HOME> v <AWAY> using the latest outputs/predictions/*.csv and output the message only." \
  --announce \
  --channel telegram \
  --to "<YOUR_CHAT_ID>"
```

---

## Troubleshooting Checklist

- Odds API errors:
  - Confirm `.env` has `ODDS_API_KEY`.
  - Run `python tipping_advisor.py --auto` manually and read the error output.
- Missing historical Parquet:
  - You need at least `data/processed/matches.parquet`, `data/processed/ladders.parquet`, `data/processed/odds.parquet`.
  - These files are ignored by git; you must generate them (or sync them) on the machine running OpenClaw.
- Cron jobs not firing:
  - Gateway must be running continuously (cron runs inside the Gateway).
  - Check cron list/runs: `openclaw cron list`, `openclaw cron runs --id <jobId>`.
- Telegram delivery goes to the wrong place:
  - Use an explicit `--to` chat id; for forum topics use `-100...:topic:<id>`.

---

## Appendix: Manual Mode (If You Don't Want Auto)

If you have your own external model pipeline:

1. Edit `tipping_advisor.py`:
   - Fill `ROUND_DATA` with the 8 matches + odds.
   - Fill `MODEL_PREDICTIONS` with `home_win_probability` for games you want the model to decide.
2. Run:

```powershell
python tipping_advisor.py
```

This path is 100% offline and does not use The Odds API.
