---
name: nrl-tuesday-tips
description: "Run the NRL Tuesday tipping pipeline: generate predictions, sanity-check, submit to ESPN Footytips, verify, and send Telegram confirmation. Diagnose and fix errors autonomously. Use for Tuesday automated tips and incident response."
---

# NRL Tuesday Tipping Pipeline

Run every Tuesday to generate NRL predictions and submit tips to ESPN Footytips. **It is critical that tips are submitted correctly — this is a real tipping competition.**

## Working directory

All commands run from: `/root/NRL-Predict`

Python is at: `.venv/bin/python3`

## Pipeline steps

Execute these steps **in order**. If any step fails, diagnose and fix the error before continuing. Do NOT skip steps.

### Step 0: Check week schedule

Before anything else, check if there's a tip deadline warning:

```bash
cd /root/NRL-Predict && cat config/week_schedule.json 2>/dev/null | python3 -m json.tool | head -20
```

Look at `tip_warning`, `first_kickoff`, and `tip_deadline`. Key scenarios:
- **`tip_warning` is set**: First kickoff is before Tuesday 5pm. Some games may already be locked. Submit remaining unlocked games ASAP.
- **First kickoff has passed**: Check which events still have `eventStatus: "pre"` (unlocked). Only those can be tipped.
- **Normal week**: First kickoff is Thursday or later. Proceed normally.

### Step 1: Generate predictions

```bash
cd /root/NRL-Predict && .venv/bin/python3 predict_round.py --auto --retrain
```

This fetches live odds from The Odds API, retrains the CatBoost model on all historical data, and writes predictions to `outputs/predictions/round_N_2026.csv`.

**Check output for:**
- "Prediction set: 8 matches" — must be 8 (or 7 during bye rounds)
- "Predictions saved to ..." — file must be written
- No Python tracebacks

**If it fails:**
- `ODDS_API_KEY` errors → check `.env` file
- Import errors → activate venv: `source .venv/bin/activate`
- "No upcoming events" → season may not have started, or API is down. Check: `.venv/bin/python3 -c "from scraping.odds_api import get_events; print(get_events())"`

### Step 2: Sanity-check predictions

Read the generated CSV and verify the blend is consistent:

```bash
cd /root/NRL-Predict && .venv/bin/python3 -c "
import pandas as pd, numpy as np
preds = pd.read_csv(sorted(__import__('pathlib').Path('outputs/predictions').glob('round_*_2026.csv'))[-1])
errors = []
for _, r in preds.iterrows():
    expected = 0.495 * r['model_CAT_top50'] + 0.505 * r['odds_home_prob']
    if abs(expected - r['home_win_prob']) > 0.02:
        errors.append(f'{r[\"home_team\"]} v {r[\"away_team\"]}: blend={r[\"home_win_prob\"]:.4f} expected={expected:.4f}')
    odds_fav_home = r['odds_home_prob'] > 0.5
    tip_home = r['tip'] == r['home_team']
    blend_home = r['home_win_prob'] > 0.5
    if tip_home != blend_home:
        errors.append(f'{r[\"home_team\"]} v {r[\"away_team\"]}: tip={r[\"tip\"]} but blend says {\"home\" if blend_home else \"away\"}')
if errors:
    print('ERRORS FOUND:'); [print(f'  {e}') for e in errors]
else:
    print(f'OK: {len(preds)} predictions verified, blend consistent')
"
```

**If errors are found:**
- Re-score from the cached model: run the scoring fix from `predict_round.py`'s `score_with_models()` function
- Regenerate the CSV with correct blend values
- **Do NOT submit tips until this check passes**

### Step 3: Ensure ESPN token is valid

```bash
cd /root/NRL-Predict && .venv/bin/python3 scripts/footytips_auth.py --check
```

If the token is expired or expiring soon (<2h):

```bash
cd /root/NRL-Predict && .venv/bin/python3 scripts/footytips_auth.py
```

**If auth fails:**
- Check `config/.footytips_creds.json` exists
- Token refresh uses Disney OneID OTP via Gmail — check Gmail access: `.venv/bin/python3 -c "from scripts.footytips_auth import check_token_expiry; print(check_token_expiry())"`
- As last resort, notify via Telegram that manual auth is needed

### Step 4: Submit tips to ESPN Footytips

First, detect the round number:

```bash
ls -1 outputs/predictions/round_*_2026.csv | sort -t_ -k2 -n | tail -1 | sed 's/.*round_\([0-9]*\)_.*/\1/'
```

Then do a **dry run** first:

```bash
cd /root/NRL-Predict && .venv/bin/python3 scripts/footytips_submit.py --round ROUND --dry-run
```

Verify the dry-run output shows the correct tips (matches the CSV). Then submit:

```bash
cd /root/NRL-Predict && .venv/bin/python3 scripts/footytips_submit.py --round ROUND
```

**Check for:**
- "Your tips have been submitted successfully"
- No HTTP errors (401 = token expired, 403 = lockout passed)

### Step 5: Verify tips via API

Confirm the submitted tips match the model predictions:

```bash
cd /root/NRL-Predict && .venv/bin/python3 -c "
import sys, pandas as pd
sys.path.insert(0, '.')
from scripts.footytips_submit import load_creds, get_auth_header, get_round_data, TEAM_NAME_BY_ID

creds = load_creds()
headers = get_auth_header(creds)
data = get_round_data(ROUND, headers)  # Replace ROUND with actual number

submitted = {TEAM_NAME_BY_ID.get(t['teamId'], '?') for t in data.get('tips', [])}
preds = pd.read_csv(sorted(__import__('pathlib').Path('outputs/predictions').glob('round_*_2026.csv'))[-1])
expected = set(preds['tip'].values)

if submitted == expected:
    print(f'VERIFIED: {len(submitted)} tips match predictions')
else:
    print('MISMATCH!')
    print(f'  Submitted: {submitted}')
    print(f'  Expected:  {expected}')
    print(f'  Missing:   {expected - submitted}')
    print(f'  Extra:     {submitted - expected}')
"
```

**If mismatch:**
- Re-submit the correct tips: `.venv/bin/python3 scripts/footytips_submit.py --round ROUND`
- If a specific game is wrong, use `--game N` to resubmit just that game
- Verify again after resubmission

### Step 6: Send Telegram notification

```bash
cd /root/NRL-Predict && .venv/bin/python3 scripts/telegram_notify.py submit ROUND N_TIPS
```

Replace ROUND with the round number and N_TIPS with the count of submitted tips (usually 8).

## Error recovery

If any step fails and you cannot fix it automatically:

1. Send a Telegram error alert:
   ```bash
   cd /root/NRL-Predict && .venv/bin/python3 scripts/telegram_notify.py error "tuesday-tips-agent" logs/pi-cron-tips-latest.log
   ```

2. Write a summary of what failed and what was attempted to `logs/pi-cron-tips-latest.md`

## Important notes

- The tipping competition is real — incorrect tips cost points
- Always dry-run before submitting
- Always verify after submitting
- The blend formula is: `home_win_prob = 0.495 * model_CAT_top50 + 0.505 * odds_home_prob`
- Tips can be resubmitted (overwritten) until lockout (first game kickoff)
- Round 1 2026 starts March 1. Regular season is rounds 1-27.
