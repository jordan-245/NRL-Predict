# NRL Match Prediction Pipeline — Agent Operations Guide

## Overview

This pipeline predicts NRL (National Rugby League) match winners using a V3 OptBlend ensemble model. It blends 7 machine learning models with bookmaker odds to achieve **68.4% accuracy** (backtested 2018–2025), beating raw bookmaker odds on all metrics.

**Model**: V3 OptBlend — weighted blend of XGBoost, LightGBM, CatBoost, Logistic Regression (each with full and top-50-feature variants) plus bookmaker implied probability.

**Data**: 2,565 historical matches (2013–2025), 131 engineered features (Elo ratings, rolling form, ladder position, head-to-head, venue stats, odds-derived features, momentum indicators).

---

## Quick Reference

```bash
# Predict all games in the next round (first run trains models, ~17s)
python predict_round.py --auto

# Predict a single game with latest odds (~4s, uses cached models)
python predict_round.py --auto --match "Sharks"

# Force retrain models (after scraping new results)
python predict_round.py --auto --retrain

# Update historical data after a round completes
python run_scrape.py
```

---

## Commands

### `predict_round.py`

The main prediction script. Fetches fixtures and odds, builds features, trains models, and outputs match predictions.

| Flag | Description |
|------|-------------|
| `--auto` | Fetch fixtures + odds from The Odds API (required for automated mode) |
| `--match "TEAM"` | Filter output to a single match. Substring match on team names, case-insensitive. Examples: `"Sharks"`, `"Storm"`, `"Panthers"` |
| `--round N` | Specify round number. Auto-detected when using `--auto`. Required for manual CSV mode |
| `--retrain` | Force full model retraining, ignoring any cached models |
| `--retune-elo` | Re-run Elo hyperparameter optimization (50 Optuna trials, adds ~30s) |
| `--year YYYY` | Season year (default: 2026) |
| `--input FILE` | Path to manual CSV file (only for non-auto mode) |

**Usage examples:**

```bash
# Auto mode — full round prediction
python predict_round.py --auto

# Auto mode — single game with freshest odds
python predict_round.py --auto --match "Sharks"

# Auto mode — specific round
python predict_round.py --auto --round 5

# Auto mode — force retrain after new results
python predict_round.py --auto --retrain

# Manual CSV mode (no API needed)
python predict_round.py --round 1
python predict_round.py --round 5 --input path/to/fixtures.csv
```

### `run_scrape.py`

Scrapes completed match results and ladder data from Rugby League Project (RLP). Updates the historical dataset that models train on.

```bash
python run_scrape.py
```

- Scrapes all rounds for seasons 2013–2026
- Caches HTML locally (only fetches new/missing pages)
- Rate-limited: 2.5s between requests (respects robots.txt)
- Outputs: `data/processed/matches.parquet`, `data/processed/ladders.parquet`, `data/processed/odds.parquet`
- First run takes ~15 minutes (fetching all HTML). Subsequent runs only fetch new rounds (~10s).

---

## Model Caching (Fast Mode)

The pipeline caches trained models after the first full run each round. Subsequent runs skip all feature engineering and model training, using cached models with fresh odds from the API.

| Mode | What happens | Time | API cost |
|------|-------------|------|----------|
| **Full path** | Load data, build features, train 7 models, predict, save cache | ~17s | 1 credit |
| **Fast path** | Load cache, fetch fresh odds, re-score | ~4s | 1 credit |

**Cache location:** `outputs/model_cache/v3_round_N_YYYY.joblib`

**Cache invalidation:** Automatic. The cache stores the modification time of `matches.parquet`. When `run_scrape.py` updates the match data, the next prediction run detects the change and does a full retrain.

**Manual invalidation:** Use `--retrain` flag to force a full retrain.

---

## Recommended Per-Game Workflow

For maximum accuracy, predict each game 1–2 hours before kickoff to capture the latest odds movements (late team changes, injury news, market shifts).

### Before Each Game (~1–2 hours pre-kickoff)

```bash
python predict_round.py --auto --match "TeamName"
```

- First game of the round: full path (~17s), builds cache
- Every subsequent game: fast path (~4s), fresh odds only

The `--match` filter accepts any substring of a team name:

| Input | Matches |
|-------|---------|
| `"Sharks"` | Cronulla Sharks vs ... |
| `"Storm"` | Melbourne Storm vs ... |
| `"Bulldogs"` | Canterbury Bulldogs vs ... |
| `"Warriors"` | New Zealand Warriors vs ... |
| `"Roosters"` | Sydney Roosters vs ... |
| `"Panthers"` | Penrith Panthers vs ... |
| `"Broncos"` | Brisbane Broncos vs ... |
| `"Raiders"` | Canberra Raiders vs ... |
| `"Cowboys"` | North Queensland Cowboys vs ... |
| `"Eels"` | Parramatta Eels vs ... |
| `"Knights"` | Newcastle Knights vs ... |
| `"Titans"` | Gold Coast Titans vs ... |
| `"Sea Eagles"` | Manly Sea Eagles vs ... |
| `"Rabbitohs"` | South Sydney Rabbitohs vs ... |
| `"Dragons"` | St George Illawarra Dragons vs ... |
| `"Tigers"` | Wests Tigers vs ... |
| `"Dolphins"` | Dolphins vs ... |

### After Each Round Completes

```bash
# 1. Scrape new results from RLP
python run_scrape.py

# 2. Next prediction auto-detects stale cache and retrains
python predict_round.py --auto
```

The scrape updates `matches.parquet`, which invalidates the model cache. The next prediction run automatically does a full retrain with the new data.

---

## Weekly Schedule

A typical NRL round runs Thursday to Monday:

| Day | Action |
|-----|--------|
| **Monday/Tuesday** (after previous round) | Run `python run_scrape.py` to update results |
| **Wednesday** (team lists announced) | Run `python predict_round.py --auto` — full round preview + builds cache |
| **Thursday** (~1h before Game 1) | `python predict_round.py --auto --match "Team1"` — fresh odds |
| **Friday** (~1h before Game 2) | `python predict_round.py --auto --match "Team2"` — fresh odds |
| **Saturday** (~1h before each game) | `python predict_round.py --auto --match "TeamN"` — fresh odds |
| **Sunday** (~1h before each game) | `python predict_round.py --auto --match "TeamN"` — fresh odds |
| **Monday** (if Monday game) | `python predict_round.py --auto --match "TeamN"` — fresh odds |

---

## Output Format

### Console Output

```
======================================================================
  NRL 2026 - ROUND 1 PREDICTIONS
  Model: V3 OptBlend (68.4% accuracy / 0.5977 log loss)
======================================================================

  Cronulla Sharks (73.1%)  vs  Gold Coast Titans (26.9%)
  >>> TIP: Cronulla Sharks                Confidence: VERY HIGH
      Odds implied: 71.1%  |  Model: 73.1%  |  Edge: +2.0%

----------------------------------------------------------------------
  TIPS SUMMARY
----------------------------------------------------------------------
  Cronulla Sharks                vs Gold Coast Titans              -> Cronulla Sharks (73%)
======================================================================
```

### Confidence Levels

| Level | Threshold | Meaning |
|-------|-----------|---------|
| VERY HIGH | > 70% | Strong edge, high conviction |
| HIGH | 60–70% | Clear favourite, solid pick |
| MEDIUM | 55–60% | Moderate lean |
| LOW | 50–55% | Close to coin flip |

### Edge Indicator

Shown when the model disagrees with bookmaker odds by more than 2%:
```
Odds implied: 71.1%  |  Model: 73.1%  |  Edge: +2.0%
```
Positive edge = model rates team higher than bookmakers (potential value bet).

### CSV Output

Predictions are saved to `outputs/predictions/round_N_YYYY.csv` with columns:

| Column | Description |
|--------|-------------|
| `home_team` | Home team (canonical name) |
| `away_team` | Away team (canonical name) |
| `home_win_prob` | Model's home win probability (0–1) |
| `away_win_prob` | Model's away win probability (0–1) |
| `odds_home_prob` | Bookmaker implied home probability |
| `tip` | Predicted winner |
| `confidence` | Confidence score (0 = coin flip, 1 = certain) |
| `model_XGBoost`, etc. | Individual model predictions |

---

## The Odds API

**Provider:** [The Odds API](https://the-odds-api.com) (v4)
**Sport key:** `rugbyleague_nrl`
**API key:** Stored in `.env` as `ODDS_API_KEY`

### Credit Usage

| Endpoint | Cost | Used for |
|----------|------|----------|
| Events (fixtures) | FREE | Getting upcoming match list |
| Odds (h2h) | 1 credit | Getting head-to-head decimal odds |
| Scores | 2 credits | Not used in standard workflow |

**Per prediction run:** 1 credit (events are free, odds cost 1)
**Monthly quota:** 500 credits on free tier
**Per round (8 games):** 8 credits max if predicting each game individually
**Per season (27 rounds):** ~216 credits, well within free tier

API quota is logged in output:
```
[Odds API] Quota: 5 used, 495 remaining
```

### Off-Season Behavior

During the NRL off-season (October–February), the API returns no events. The script will print an error:
```
ERROR: No upcoming NRL events found.
  The season may not have started yet, or it may be the off-season.
  Use manual CSV mode instead: python predict_round.py --round N
```

---

## Team Names

The pipeline uses canonical NRL team names. All inputs (API, CSV, scraping) are standardised via `config/team_mappings.py`.

| Canonical Name | Common Aliases |
|---------------|----------------|
| Brisbane Broncos | Broncos |
| Canberra Raiders | Raiders |
| Canterbury Bulldogs | Bulldogs, Canterbury-Bankstown Bulldogs |
| Cronulla Sharks | Sharks, Cronulla Sutherland Sharks |
| Dolphins | Redcliffe Dolphins |
| Gold Coast Titans | Titans |
| Manly Sea Eagles | Sea Eagles, Manly Warringah Sea Eagles |
| Melbourne Storm | Storm |
| New Zealand Warriors | Warriors, NZ Warriors |
| Newcastle Knights | Knights |
| North Queensland Cowboys | Cowboys, NQ Cowboys |
| Parramatta Eels | Eels |
| Penrith Panthers | Panthers |
| South Sydney Rabbitohs | Rabbitohs, Souths |
| St George Illawarra Dragons | Dragons, SGI Dragons |
| Sydney Roosters | Roosters, Eastern Suburbs Roosters |
| Wests Tigers | Tigers, West Tigers |

---

## File Structure

```
NRL-Predict/
├── predict_round.py          # Main prediction script
├── run_scrape.py             # Data scraping pipeline
├── run_enhance_and_retrain.py # V3 feature engineering functions (imported by predict_round.py)
├── .env                      # API keys (ODDS_API_KEY)
├── config/
│   ├── settings.py           # Global config (paths, years, scraping params)
│   ├── team_mappings.py      # Team name standardisation
│   └── elo_params.json       # Cached Elo hyperparameters
├── scraping/
│   ├── odds_api.py           # The Odds API v4 client
│   ├── rlp_scraper.py        # Rugby League Project HTML scraper
│   └── odds_loader.py        # Historical odds Excel parser
├── data/
│   ├── processed/            # Parquet files (matches, ladders, odds)
│   ├── raw/                  # Cached HTML and source files
│   └── upcoming/             # Manual CSV fixtures (for non-auto mode)
├── outputs/
│   ├── predictions/          # Round prediction CSVs
│   └── model_cache/          # Cached trained models (.joblib)
└── PIPELINE_GUIDE.md         # This file
```

---

## Troubleshooting

### "ODDS_API_KEY not set"
Add your API key to `.env`:
```
ODDS_API_KEY=your_key_here
```

### "No upcoming NRL events found"
The season hasn't started or it's the off-season. Use manual CSV mode:
```bash
python predict_round.py --round N
```

### "Cache stale (new results detected)"
This is normal — it means `run_scrape.py` updated the data and models will retrain. No action needed.

### Predictions seem wrong or unchanged
Force a full retrain:
```bash
python predict_round.py --auto --retrain
```

### API quota running low
Check remaining credits in output (`[Odds API] Quota: X used, Y remaining`). The free tier resets monthly. If running low, reduce per-game predictions and do full-round predictions instead (1 credit per round vs 8 per round).

### Unknown team name from API
Check the output for warnings like:
```
Unrecognised team from Odds API: 'Team Name'
```
Add the team name to `TEAM_ALIASES` in `config/team_mappings.py`.

---

## Model Details

### V3 OptBlend Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| XGBoost (131 features) | 1.025 | Gradient boosted trees, full feature set |
| LightGBM (131 features) | -0.599 | Light gradient boosting, full features |
| CatBoost (131 features) | 0.281 | Categorical boosting, full features |
| LogReg (131 features) | 0.054 | Regularised logistic regression |
| XGBoost (top 50) | -1.126 | XGBoost on 50 most important features |
| LightGBM (top 50) | 0.540 | LightGBM on 50 most important features |
| CatBoost (top 50) | 0.005 | CatBoost on 50 most important features |
| Bookmaker Odds | 0.819 | Implied probability from h2h odds |

Negative weights indicate the model contributes by *contrasting* with other models (ensemble diversification).

### Feature Categories (131 total)

| Category | Count | Examples |
|----------|-------|---------|
| Elo ratings | 4 | home_elo, away_elo, elo_diff, home_elo_prob |
| Rolling form (3/5/8 game) | 30 | win_rate, avg_points_for/against, avg_margin |
| Ladder position | 10 | ladder_pos, wins, losses, competition_points |
| Ladder home/away splits | 16 | home_team_home_win_pct, away_team_away_ppg |
| Schedule | 7 | days_rest, back_to_back, bye_last_round |
| Context | 5 | is_home, round_number, is_finals, day_of_week |
| Odds-derived | 12 | odds_home_prob, odds_favourite, odds_movement |
| Head-to-head | 9 | h2h_win_rate, h2h_avg_margin (3/5/all games) |
| Venue | 4 | venue_win_rate, is_neutral_venue |
| Momentum | 10 | form_momentum, streak, last_result |
| Halftime/Penalty | 6 | avg_halftime_lead, avg_penalty_diff |
| Engineered interactions | 18 | elo_x_rest, odds_elo_diff, strength_diff |

### Training Configuration

- **Training data**: 2013–2025 (2,556 matches after excluding draws)
- **Sample weighting**: Exponential decay 0.9^(max_year - year), favouring recent seasons
- **Elo tuning**: Optuna 50 trials, cached in `config/elo_params.json`
- **XGBoost**: n_estimators=204, max_depth=3, learning_rate=0.0148
- **LightGBM**: n_estimators=459, num_leaves=22, max_depth=2, learning_rate=0.0096
- **CatBoost**: iterations=439, depth=4, learning_rate=0.0106

### Backtested Performance (Walk-Forward 2018–2025)

| Metric | V3 OptBlend | Bookmaker Odds |
|--------|-------------|----------------|
| Accuracy | **68.4%** | 68.0% |
| Log Loss | **0.5977** | 0.6000 |
