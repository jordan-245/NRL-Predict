# NRL Match Winner Prediction Model — Project Plan

## 1. Project Overview

Build a machine learning pipeline to predict the winning team of each NRL match in the 2026 season, trained and backtested on historical data from 2013–2025. The primary data source is [Rugby League Project](https://www.rugbyleagueproject.org/) (RLP), supplemented by the [AusSportsBetting historical odds dataset](https://www.aussportsbetting.com/data/historical-nrl-results-and-odds-data/) and optionally the NRL official site via the [beauhobba/NRL-Data scraper](https://github.com/beauhobba/NRL-Data).

---

## 2. Data Sources

### 2.1 Rugby League Project (Primary — Scrape)

RLP is a static HTML site with a consistent URL scheme. No API exists; data must be scraped.

#### URL Patterns

| Entity | URL Template | Notes |
|---|---|---|
| Season summary | `/seasons/nrl-{year}/summary.html` | Teams, games count, top scorers, attendance stats |
| Season results (all matches) | `/seasons/nrl-{year}/results.html` | Full list of every match in the season |
| Round results | `/seasons/nrl-{year}/round-{N}/summary.html` | Per-match detail including lineups |
| Round ladder | `/seasons/nrl-{year}/round-{N}/ladder.html` | Standings after each round |
| Match summary | `/seasons/nrl-{year}/round-{N}/{home}-vs-{away}/summary.html` | Scorers, teams, venue, crowd, referee, lineups, halftime score |
| Match stats | `/seasons/nrl-{year}/round-{N}/{home}-vs-{away}/stats.html` | Detailed match statistics (where available) |
| Season players | `/seasons/nrl-{year}/players.html` | All players who appeared that season with aggregate stats |
| Player profile | `/players/{player-slug}/summary.html` | Career bio, season-by-season appearances, scoring |
| Competition teams | `/competitions/nrl/teams.html` | All-time team list |
| Data completeness | `/seasons/nrl-{year}/data.html` | Shows which data columns are available per season |
| Ladder predictor | `/seasons/nrl-{year}/ladder.html` | End-of-season ladder |
| Season venues | `/seasons/nrl-{year}/venues.html` | Venues used that season |
| Finals rounds | `/seasons/nrl-{year}/qualif-final/summary.html`, `elim-final`, `semi-final`, `prelim-final`, `grand-final` | Same structure as regular rounds |

#### Team Slug Convention

Team slugs in match URLs use lowercase-hyphenated full names, e.g.:
`melbourne-storm-vs-penrith-panthers`, `south-sydney-rabbitohs-vs-canterbury-bankstown-bulldogs`

#### Data Available Per Match (from Round Pages)

From the round summary pages, each match row contains:

- **Home team** and **away team** names
- **Final score** (home and away)
- **Try scorers** with individual try counts per player
- **Goal kickers** with goal counts
- **Field goals** (if any)
- **Venue** name
- **Date** and **kickoff time**
- **Halftime score**
- **Penalty count** (home vs away)
- **Referee**
- **Attendance** (crowd figure)
- **Full team lineups**: starting 13 (numbered 1–13 by position: fullback, wing, centre, five-eighth, halfback, prop, hooker, second-row, lock) plus interchange bench (typically 4 players)
- **Team changes** from previous game: ins, outs, positional moves

#### Data Available Per Player

- Career biography (free text)
- Season-by-season breakdown: appearances, tries, goals, field goals, points
- Position(s) played
- Teams represented
- Win/loss record (filterable via `w%` parameter)

### 2.2 AusSportsBetting (Supplementary — Download)

Free Excel download at `https://www.aussportsbetting.com/data/historical-nrl-results-and-odds-data/`

Fields available:

- Date, kickoff (local time)
- Home team, away team
- Home score, away score
- Playoff game flag (Y/N)
- Head-to-head odds: home, draw, away
- From 2013 onward: opening, minimum, maximum, and closing odds for head-to-head, line (handicap), and total score markets
- Bookmakers surveyed count

This provides a strong market-implied probability baseline to benchmark model performance against.

### 2.3 NRL Official Site / beauhobba/NRL-Data (Optional Enhancement)

The [beauhobba/NRL-Data](https://github.com/beauhobba/NRL-Data) GitHub project scrapes the official NRL website and provides:

- In-depth match statistics: possession %, completion rate, run metres, tackles, errors, offloads, line breaks, kick metres, etc.
- Individual player stats per match: tackles, runs, run metres, try assists, line break assists, etc.
- Data hosted at a public S3 endpoint in JSON format

These advanced match-level and player-level stats (not available on RLP) are extremely valuable features for prediction.

### 2.4 Weather Data (Optional Enhancement)

Historical weather at match time/venue from the [Bureau of Meteorology](http://www.bom.gov.au/) or [Open-Meteo API](https://open-meteo.com/). Useful as rain and wind affect scoring patterns.

---

## 3. Project Directory Structure

```
nrl-predictor/
├── README.md
├── pyproject.toml                  # Project deps (or requirements.txt)
├── .env.example                    # Config template
│
├── config/
│   ├── settings.py                 # Global config (paths, date ranges, team mappings)
│   ├── team_mappings.py            # Canonical team name → aliases mapping
│   └── feature_config.yaml         # Feature engineering toggle config
│
├── scraping/
│   ├── __init__.py
│   ├── rlp_scraper.py              # Core RLP HTML scraper
│   ├── rlp_match_parser.py         # Parse match detail from round pages
│   ├── rlp_ladder_parser.py        # Parse ladder tables
│   ├── rlp_player_parser.py        # Parse player profiles & season player lists
│   ├── rlp_url_builder.py          # URL generation helpers
│   ├── odds_loader.py              # Load AusSportsBetting Excel file
│   ├── nrl_stats_loader.py         # (Optional) Load beauhobba/NRL-Data JSON
│   ├── weather_loader.py           # (Optional) Weather API integration
│   ├── rate_limiter.py             # Polite scraping: delays, retries, caching
│   └── cache/                      # Local HTML cache directory
│       └── .gitkeep
│
├── data/
│   ├── raw/                        # Raw scraped/downloaded data
│   │   ├── rlp/                    # Cached HTML files by season/round
│   │   ├── odds/                   # AusSportsBetting Excel files
│   │   └── nrl_stats/              # beauhobba JSON files
│   ├── processed/                  # Cleaned, parsed tabular data
│   │   ├── matches.parquet         # Master match table
│   │   ├── lineups.parquet         # Match-level team lineups
│   │   ├── players.parquet         # Player career/season stats
│   │   ├── ladders.parquet         # Round-by-round ladder snapshots
│   │   ├── odds.parquet            # Bookmaker odds per match
│   │   └── match_stats.parquet     # (Optional) Advanced match stats
│   └── features/                   # Engineered feature sets ready for modelling
│       ├── features_v1.parquet
│       └── features_v2.parquet
│
├── processing/
│   ├── __init__.py
│   ├── data_cleaning.py            # Standardise team names, dates, handle nulls
│   ├── data_linking.py             # Join matches ↔ odds ↔ lineups ↔ stats
│   ├── feature_engineering.py      # Feature computation (see §5)
│   ├── elo.py                      # Elo/Glicko rating system
│   ├── rolling_stats.py            # Rolling window team/player aggregations
│   └── target_encoding.py          # Target variable creation
│
├── modelling/
│   ├── __init__.py
│   ├── baseline_models.py          # Home-team, odds-implied, Elo baselines
│   ├── classical_models.py         # Logistic Regression, Random Forest, XGBoost, LightGBM
│   ├── neural_models.py            # MLP, LSTM/GRU sequence models
│   ├── ensemble.py                 # Stacking, blending, voting ensembles
│   ├── hyperparameter_search.py    # Optuna / sklearn search wrappers
│   ├── calibration.py              # Probability calibration (Platt, isotonic)
│   ├── model_registry.py           # Save/load/version models
│   └── interpretability.py         # SHAP, feature importance, partial dependence
│
├── evaluation/
│   ├── __init__.py
│   ├── backtesting.py              # Walk-forward / expanding window backtester
│   ├── metrics.py                  # Accuracy, log-loss, Brier score, AUC, calibration
│   ├── betting_simulation.py       # Simulated P&L against closing odds
│   └── reports.py                  # Generate evaluation reports & plots
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_deep_dive_errors.ipynb
│   └── 05_2026_predictions.ipynb
│
├── pipelines/
│   ├── scrape_all.py               # End-to-end scrape orchestrator
│   ├── build_features.py           # Raw → processed → features
│   ├── train_and_evaluate.py       # Full training + backtest pipeline
│   └── predict_upcoming.py         # Generate predictions for upcoming rounds
│
├── tests/
│   ├── test_scraper.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_backtest.py
│
└── outputs/
    ├── models/                     # Serialised trained models
    ├── predictions/                # CSV/JSON prediction outputs
    ├── reports/                    # HTML/PDF evaluation reports
    └── figures/                    # Charts and plots
```

---

## 4. Data Extraction Plan

### 4.1 Scraping Strategy for RLP

RLP is a volunteer-run site — scrape politely.

**Principles:**
- Cache every HTML page locally on first fetch; never re-fetch cached pages
- Rate limit: 1 request per 2–3 seconds minimum
- Set a descriptive `User-Agent` string
- Respect `robots.txt`
- Scrape once in bulk for historical data; only scrape new rounds incrementally

**Scrape Order:**

1. **Season list**: hardcode NRL seasons 2013–2026 (URL: `/seasons/nrl-{year}/summary.html`)
2. **Round-by-round results**: for each season, iterate rounds 1–27 (or whatever the max is) plus finals rounds (`qualif-final`, `elim-final`, `semi-final`, `prelim-final`, `grand-final`). Fetch `/seasons/nrl-{year}/round-{N}/summary.html`. This single page per round gives you every match with full detail (scores, lineups, venue, crowd, referee, halftime score, penalties).
3. **Round ladders**: fetch `/seasons/nrl-{year}/round-{N}/ladder.html` for every round. Gives win/loss/draw/for/against/points at that point in the season.
4. **Match stats pages** (where available): fetch `/seasons/nrl-{year}/round-{N}/{matchslug}/stats.html` for advanced per-match stats. Check the data completeness page first to know which seasons have this.
5. **Season player lists**: fetch `/seasons/nrl-{year}/players.html` for a per-season player index with aggregate stats (appearances, tries, goals, points).
6. **Player profiles** (selective): for key players or to build position-level features, fetch `/players/{slug}/summary.html`. Can be done lazily — only fetch players who appear in the lineups data.

**Parsing Approach:**

Use `BeautifulSoup` or `selectolax`/`lxml` for HTML parsing. The round summary pages contain match data in a structured text format:

```
> {Home} {HomeScore} ({scorers}) defeated/drew/lost to {Away} {AwayScore} ({scorers})
  at {Venue}. Date: {Day}, {Date}. Kickoff: {Time}.
  Halftime: {HalfScore}. Penalties: {HomePen}-{AwayPen}.
  Referee: {Ref}. Crowd: {Crowd}.
  {Home}: {Fullback}, {Wing}, {Centre}, ... Int: {Bench1}, {Bench2}, ...
  {Away}: {Fullback}, {Wing}, ...
```

Build regex and/or BeautifulSoup parsing for this repeating pattern. Note the `>` link at the start takes you to the individual match page.

### 4.2 Loading AusSportsBetting Odds

- Download the NRL Excel file manually from `https://www.aussportsbetting.com/data/historical-nrl-results-and-odds-data/`
- Parse with `openpyxl` or `pandas.read_excel()`
- Fields: date, home team, away team, scores, playoff flag, home/draw/away odds (open/min/max/close from 2013+)
- Join to RLP match data on (date, home_team, away_team) after standardising team names

### 4.3 Team Name Standardisation

Critical because different sources use different team names. Build a canonical mapping:

```python
TEAM_ALIASES = {
    "Penrith Panthers":         ["Penrith", "Panthers"],
    "Melbourne Storm":          ["Melbourne", "Storm"],
    "Sydney Roosters":          ["Sydney", "Roosters", "Eastern Suburbs"],
    "South Sydney Rabbitohs":   ["South Sydney", "Souths", "Rabbitohs"],
    "Canterbury Bulldogs":      ["Canterbury", "Canterbury-Bankstown Bulldogs", "Bulldogs", "CBY"],
    "Cronulla Sharks":          ["Cronulla", "Cronulla-Sutherland Sharks", "Sharks"],
    "St George Illawarra":      ["St Geo Illa", "St George Illawarra Dragons", "Dragons"],
    "Manly Sea Eagles":         ["Manly", "Manly-Warringah Sea Eagles", "Sea Eagles"],
    "New Zealand Warriors":     ["NZ Warriors", "Warriors", "New Zealand"],
    "North Queensland Cowboys": ["North Qld", "North QLD Cowboys", "Cowboys"],
    "Parramatta Eels":          ["Parramatta", "Eels"],
    "Brisbane Broncos":         ["Brisbane", "Broncos"],
    "Newcastle Knights":        ["Newcastle", "Knights"],
    "Canberra Raiders":         ["Canberra", "Raiders"],
    "Wests Tigers":             ["Wests Tigers", "Tigers"],
    "Gold Coast Titans":        ["Gold Coast", "Titans"],
    "Dolphins":                 ["Dolphins", "Redcliffe Dolphins", "DOL"],
}
```

### 4.4 Data Volume Estimate

- ~13 seasons (2013–2025) × ~200 matches/season = ~2,600 matches
- ~2,600 matches × ~34 players per match (17 per side) = ~88,000 lineup rows
- ~27 rounds × 13 seasons × 17 teams = ~6,000 ladder snapshots
- Manageable dataset — fits comfortably in memory

---

## 5. Feature Engineering

### 5.1 Team-Level Features (Per Match)

**Recent Form (Rolling Windows):**
- Win rate over last N games (N = 3, 5, 8)
- Points scored / conceded rolling average
- Points differential rolling average
- Try scoring rate (last 5)
- Completion rate trend (if advanced stats available)

**Elo / Power Rating:**
- Implement Elo rating system with tunable K-factor
- Also test Glicko-2 (includes rating volatility)
- Rate each team entering each match — strong single-feature predictor

**Season Standing:**
- Current ladder position
- Win %, points differential, for/against ratio as of pre-match
- Games behind the leader

**Head-to-Head:**
- Historical win rate of Team A vs Team B (all-time, last 3 years, last 5 meetings)
- Average margin in recent H2H meetings

**Home/Away:**
- Is the team playing at home?
- Home win rate this season
- Away win rate this season
- Historical home/away performance at this specific venue

**Schedule / Fatigue:**
- Days since last game (short turnovers = disadvantage)
- Is this a back-to-back game within 5 days?
- Travel distance from last game venue to this venue (derive from venue geolocation lookup)
- Bye-week effect (did the team have a bye the previous round?)
- Number of games in last 14/21 days

### 5.2 Player-Level Features (Aggregated to Team)

**Lineup Strength:**
- Sum of career appearances of starting 13 (proxy for experience)
- Sum of career tries / points of starting 13
- Average "games this season" of starting lineup (form/fitness proxy)
- Number of debutants or low-experience players in lineup
- Number of changes from previous round's lineup (stability proxy)

**Key Player Availability:**
- Is the #7 (halfback) the same as last week?
- Is the #1 (fullback) the same?
- Number of positional changes (reshuffled team = disruption signal)

**Position Group Strength:**
- Aggregate stats for spine players (1, 6, 7, 9) vs forwards (8–13) vs backs (2–5)
- Experience-weighted position scores

### 5.3 Venue Features

- Venue name (one-hot or target-encoded)
- Is this a team's traditional home ground?
- Historical average margin at this venue
- Venue capacity (proxy for atmosphere/crowd pressure)
- Altitude / climate zone (minimal for NRL but not zero)

### 5.4 Contextual Features

- Round number (early season vs late season dynamics)
- Is this a finals match? (higher stakes, different dynamics)
- Day of week / time slot (Friday night, Saturday afternoon, etc.)
- Season year (to capture era effects)
- Rivalry flag (known intense rivalries)

### 5.5 Market / Odds Features (for Model Comparison, Not Leakage)

- Closing head-to-head odds → implied probability (use as benchmark, or as a feature in a blended model)
- Opening vs closing odds movement (line movement captures injury/team news information)
- Handicap line spread
- Over/under total

**Note:** When using odds as a feature, use opening odds only (available before the match) to avoid information leakage from late team news.

### 5.6 Weather Features (Optional)

- Temperature at kickoff
- Rainfall (mm) in preceding 24 hours
- Wind speed
- Humidity

---

## 6. Modelling Strategy

### 6.1 Target Variable

Binary classification: `home_win` (1 = home team wins, 0 = away team wins). Draws are extremely rare in NRL (<1% of matches due to golden point extra time) — can be handled as a separate class or excluded.

### 6.2 Baseline Models

These establish the floor performance to beat:

| Model | Description | Expected Accuracy |
|---|---|---|
| **Home always wins** | Predict home team every match | ~55–58% |
| **Ladder position** | Higher-ranked team wins | ~60% |
| **Odds-implied** | Convert closing odds to probabilities, pick favourite | ~68–72% |
| **Elo rating** | Simple Elo system, pick higher-rated team | ~63–67% |

The **odds-implied baseline (~70%)** is the main benchmark to beat. Bookmaker odds incorporate immense information — beating them consistently is the true test.

### 6.3 Classical ML Models

| Model | Why | Key Hyperparameters |
|---|---|---|
| **Logistic Regression** | Interpretable, fast, good calibration baseline | Regularisation (L1/L2), C |
| **Random Forest** | Handles non-linearities, feature interactions | n_estimators, max_depth, min_samples_leaf |
| **XGBoost** | State-of-the-art for tabular data, handles missing values | learning_rate, max_depth, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda |
| **LightGBM** | Faster than XGBoost, good with categorical features | Same family of params + categorical_feature, num_leaves |
| **CatBoost** | Native categorical handling, ordered boosting reduces overfitting | depth, iterations, learning_rate, l2_leaf_reg |
| **SVM (RBF kernel)** | Can capture non-linear boundaries | C, gamma |

### 6.4 Neural / Deep Learning Models

| Model | Rationale |
|---|---|
| **MLP (Feedforward)** | Simple NN baseline; compare vs gradient boosting |
| **LSTM / GRU** | Treat team's sequence of match results as a time series; model momentum/form as sequential patterns |
| **Transformer (tabular)** | Attention mechanism may capture feature interactions; TabTransformer or FT-Transformer |

### 6.5 Ensemble Approaches

| Method | Description |
|---|---|
| **Voting (hard/soft)** | Combine top-3 models by majority vote or averaged probability |
| **Stacking** | Train a meta-learner (logistic regression) on out-of-fold predictions from base models |
| **Blending with odds** | Blend model probability with odds-implied probability using optimised weights |

### 6.6 Hyperparameter Optimisation

Use **Optuna** with TPE (Tree-structured Parzen Estimator) sampler:

- Define search spaces per model family
- Optimise for log-loss (preferred over accuracy for probability estimation)
- Use cross-validated backtest score as the objective (see §7)
- Budget: 200–500 trials per model type
- Use pruning (MedianPruner) to early-stop poor trials

### 6.7 Probability Calibration

After training, calibrate output probabilities:

- **Platt scaling** (logistic regression on model outputs)
- **Isotonic regression** (non-parametric calibration)
- Evaluate with reliability diagrams and Expected Calibration Error (ECE)

Well-calibrated probabilities are essential for betting simulation and practical utility.

---

## 7. Backtesting Framework

### 7.1 Walk-Forward Validation

**Do not use random train/test splits** — this causes temporal leakage.

Use **expanding window** or **sliding window** walk-forward validation:

```
Train: 2013–2017  →  Test: 2018
Train: 2013–2018  →  Test: 2019
Train: 2013–2019  →  Test: 2020
Train: 2013–2020  →  Test: 2021
Train: 2013–2021  →  Test: 2022
Train: 2013–2022  →  Test: 2023
Train: 2013–2023  →  Test: 2024
Train: 2013–2024  →  Test: 2025
```

This simulates real-world deployment: only past data is used to predict future matches.

For within-season granularity, also test **round-by-round walk-forward**: train on all data up to round N, predict round N+1, expanding through the season.

### 7.2 Evaluation Metrics

| Metric | Description | Purpose |
|---|---|---|
| **Accuracy** | % of matches correctly predicted | Simple headline metric |
| **Log-loss** | Penalises confident wrong predictions | Primary optimisation target |
| **Brier score** | Mean squared error of probabilities | Measures calibration + discrimination |
| **AUC-ROC** | Discrimination ability | How well the model separates winners from losers |
| **Calibration plot** | Predicted vs actual probability | Visual check of probability reliability |
| **ECE** | Expected Calibration Error | Single-number calibration quality |

### 7.3 Betting Simulation

Simulate a betting strategy against historical bookmaker odds:

- **Flat stake value betting**: bet 1 unit on the team whenever model probability exceeds implied odds probability by a threshold (e.g., >5% edge)
- **Kelly criterion**: size bets proportional to perceived edge
- Track: total bets, win rate, ROI, max drawdown, Sharpe ratio
- Compare across model variants

This is the ultimate real-world test: does the model identify genuine value that bookmakers miss?

### 7.4 Model Comparison Framework

Run all model variants through the same backtesting harness and collect results in a standardised comparison table:

```
| Model           | Feature Set | Accuracy | Log-Loss | Brier | AUC  | Sim. ROI |
|-----------------|-------------|----------|----------|-------|------|----------|
| Home Baseline   | None        | 56.2%    | 0.689    | 0.247 | 0.50 | -8.2%    |
| Odds Implied    | Odds only   | 70.1%    | 0.582    | 0.198 | 0.74 | -4.5%    |
| Elo             | Elo only    | 64.8%    | 0.631    | 0.222 | 0.68 | -6.1%    |
| LogReg          | v1          | 66.3%    | 0.615    | ...   | ...  | ...      |
| XGBoost         | v2          | ...      | ...      | ...   | ...  | ...      |
| LightGBM        | v2          | ...      | ...      | ...   | ...  | ...      |
| LSTM            | v2+seq      | ...      | ...      | ...   | ...  | ...      |
| Stacked Ens.    | v2          | ...      | ...      | ...   | ...  | ...      |
```

---

## 8. Configuration Matrix — What to Test

### 8.1 Feature Set Variants

| Version | Features Included |
|---|---|
| v1_basic | Elo, home/away, ladder position, days rest, round number |
| v2_form | v1 + rolling form (3/5/8 game windows), H2H record, recent margin |
| v3_lineup | v2 + lineup experience, changes from previous week, key player availability |
| v4_advanced | v3 + advanced match stats (possession, completion, run metres — if available) |
| v5_odds | v4 + opening odds as features |
| v6_weather | v5 + weather at match time |

### 8.2 Elo Hyperparameters to Tune

- K-factor: test 15, 20, 25, 30, 40
- Home advantage points: test 30, 40, 50, 60
- Season reset factor: test 0.5, 0.6, 0.75, 1.0 (1.0 = no reset)
- Margin-of-victory adjustment: linear vs logarithmic vs none

### 8.3 Rolling Window Sizes

- Form windows: [3, 5, 8, 10] games
- H2H lookback: [last 3 meetings, last 5 meetings, last 3 years]
- Exponentially weighted moving averages with half-life parameter

### 8.4 Training Window Sizes

- Full history (2013–test year)
- Last 5 seasons only
- Last 3 seasons only
- Recency-weighted (all data, but exponential decay on older samples)

### 8.5 Model-Specific Configurations

**XGBoost / LightGBM:**
- max_depth: [3, 4, 5, 6, 8]
- learning_rate: [0.01, 0.03, 0.05, 0.1]
- n_estimators: [100, 300, 500, 1000]
- subsample: [0.7, 0.8, 0.9]
- colsample_bytree: [0.6, 0.7, 0.8, 1.0]
- min_child_weight: [1, 3, 5, 10]

**LSTM:**
- Hidden dimensions: [32, 64, 128]
- Sequence length: [3, 5, 8] (number of previous matches)
- Layers: [1, 2]
- Dropout: [0.1, 0.2, 0.3]
- Learning rate: [1e-4, 5e-4, 1e-3]

---

## 9. Implementation Phases

### Phase 1: Data Collection (Week 1–2)

1. Build RLP scraper with local HTML caching and rate limiting
2. Scrape all NRL round pages for 2013–2025 (and 2026 rounds as they become available)
3. Parse matches, lineups, scores, venues, crowds, referees
4. Scrape round-by-round ladder data
5. Download and parse AusSportsBetting odds Excel
6. Standardise team names across all sources
7. Join datasets on (date, home_team, away_team) → master `matches` table
8. Store as Parquet files in `data/processed/`

### Phase 2: Feature Engineering (Week 2–3)

1. Implement Elo rating system, backfill ratings for every match
2. Compute rolling form features at multiple window sizes
3. Build lineup-strength features from player career stats
4. Compute H2H features, venue features, schedule features
5. Create feature set versions (v1 through v6)
6. Exploratory data analysis notebooks — feature distributions, correlations, class balance

### Phase 3: Baseline Models (Week 3)

1. Implement home-always, ladder-position, and odds-implied baselines
2. Implement and tune Elo baseline
3. Set up walk-forward backtesting harness
4. Establish performance floor across all metrics

### Phase 4: Classical ML Models (Week 3–4)

1. Train Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
2. Run Optuna hyperparameter search for each model × each feature set
3. Evaluate all combinations via walk-forward backtest
4. Analyse feature importance (SHAP values for tree models)
5. Identify best single models and best feature sets

### Phase 5: Deep Learning & Ensembles (Week 4–5)

1. Build LSTM/GRU model treating each team's season as a sequence
2. Experiment with MLP and TabTransformer architectures
3. Build stacking ensemble from top-3 base models
4. Build blended model (model probability + odds probability)
5. Run full backtest comparison across all approaches

### Phase 6: Calibration & Final Evaluation (Week 5)

1. Calibrate probabilities of best models (Platt/isotonic)
2. Full betting simulation against historical odds
3. Generate final comparison report with all metrics
4. Error analysis: which types of matches are hardest to predict?
5. Select production model configuration

### Phase 7: 2026 Deployment (Ongoing)

1. Automate weekly scraping of new RLP round data
2. Update features incrementally
3. Generate round-by-round predictions
4. Track live accuracy through the season
5. Retrain periodically (e.g., monthly) as new 2026 data accumulates

---

## 10. Key Technical Decisions & Notes

### Why 2013 as the Start Year?

- AusSportsBetting odds data includes opening/closing/min/max odds from 2013 onward, essential for robust benchmarking
- The modern NRL (17-team competition, current rules) has been relatively stable since ~2012
- Going further back introduces team composition changes (e.g., Gold Coast Titans entry 2007, rule changes)
- 13 seasons ≈ 2,600 matches is sufficient training data for tabular ML

### Handling Draws

Draws are extremely rare in modern NRL due to golden point extra time. Options:
1. Exclude draws entirely (~<1% of matches)
2. Treat as a separate class (multiclass: home_win / draw / away_win)
3. Model as "no clear winner" — both teams get 0.5

Recommended: option 1 (exclude) for simplicity, or option 2 if you want to explore it later.

### Avoiding Data Leakage

- **Never** use end-of-season stats as features for mid-season matches
- All features must be computed from data available *before* the match in question
- Odds: use only opening odds if using as a feature (closing odds incorporate late information)
- Ladder position: use the ladder *entering* the round, not after it

### Scraping Ethics

- RLP is a volunteer project — treat their servers with respect
- Cache aggressively, scrape once
- Consider reaching out to the RLP team to ask if they have data exports available
- Attribute the data source in any publications

---

## 11. Dependencies

```
# Core
python >= 3.10
pandas >= 2.0
numpy
pyarrow                    # Parquet I/O

# Scraping
requests
beautifulsoup4
lxml
tqdm

# ML
scikit-learn
xgboost
lightgbm
catboost
optuna                     # Hyperparameter search

# Deep Learning
torch                      # PyTorch for LSTM/Transformer
pytorch-lightning          # (optional) training harness

# Evaluation & Viz
matplotlib
seaborn
shap
plotly                     # Interactive charts

# Data
openpyxl                   # Excel reading (odds data)
pyyaml                     # Config files

# Utilities
joblib                     # Model serialisation
python-dotenv              # Environment config
```

---

## 12. Success Criteria

| Tier | Goal |
|---|---|
| **Bronze** | Beat the Elo baseline (~65% accuracy) with a reproducible ML pipeline |
| **Silver** | Match or beat the odds-implied baseline (~70% accuracy, log-loss < 0.58) |
| **Gold** | Demonstrate positive simulated ROI against historical bookmaker odds over 3+ backtested seasons |
| **Platinum** | Achieve positive live ROI during the 2026 NRL season |

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| RLP site structure changes | Scraper breaks | Cache all HTML locally; build parser with fallback logic |
| Missing data for some seasons | Feature gaps | Check data completeness pages; degrade gracefully to simpler features |
| Team name mismatches between sources | Broken joins, data loss | Comprehensive alias mapping; fuzzy matching as fallback |
| Overfitting on small dataset (~2,600 matches) | Poor out-of-sample performance | Strong regularisation; walk-forward validation only; feature selection |
| Cannot beat bookmaker odds | No practical edge | Still valuable as an analysis tool; focus on sub-markets (e.g., early-season when odds are less efficient) |
| NRL rule changes across years | Non-stationary data | Use recent seasons more heavily; test sliding vs expanding windows |