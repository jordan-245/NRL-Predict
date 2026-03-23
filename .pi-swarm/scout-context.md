

### scout-1
## Scout Report: NRL-Predict Feature Engineering & Model Improvements

### Files Retrieved

1. **`pipelines/v4.py`** (lines 1-1266+) — V4 feature engineering pipeline with 12 feature computation functions following pattern `compute_X_features(matches)`. Includes new features: spread movement, scoring consistency, attendance, kickoff time, lineup stability, player impact, referee home bias, rolling match stats, and player form aggregation.

2. **`predict_round.py`** (lines 1-1387) — Prediction generation script. Contains `FEATURE_COLS` list (194 features across V3 base + V4 enhancements), model training (`train_and_predict`), and blending logic (35% CatBoost + 65% odds). Also handles lineup checks and probability adjustments.

3. **`pipelines/v3.py`** (lines 1-1650+) — Base feature pipeline (V3). Foundation for Elo, rolling form, ladder, venue, odds, and engineering features. Provides `link_odds()`, `backfill_elo()`, and 10+ `compute_X_features()` functions.

4. **`config/settings.py`** (full) — Global configuration. Defines paths, scraping parameters, season boundaries, round definitions, and random seed.

5. **`processing/player_impact.py`** (lines 1-100) — Computes player impact scores via Elo-adjusted comparison of team performance with/without each player. Requires `player_appearances.parquet` as input.

### Key Code

#### Feature Function Pattern (V3 & V4)
```python
def compute_X_features(matches: pd.DataFrame, ...) -> pd.DataFrame:
    """Compute features and return modified DataFrame.
    
    Standard pattern:
    - Copy input DataFrame
    - Build per-team or per-match lookups
    - Attach new columns using vectorized operations or loops
    - Return augmented DataFrame
    """
    df = matches.copy()
    # ... computation ...
    return df
```

#### V4 Feature Functions (12 total)
1. `compute_v4_odds_features()` — Spread movement, total line movement, draw odds competitiveness
2. `compute_scoring_consistency_features()` — Scoring std dev, close game tendency, points trend
3. `compute_attendance_features()` — Attendance normalized to team average
4. `compute_kickoff_features()` — Day/afternoon/night game flags
5. `compute_lineup_stability_features()` — Player retention between games from `player_appearances.parquet`
6. `compute_player_impact_features()` — Spine and total starter impact sums from `player_impact.parquet`
7. `compute_v4_engineered_features()` — Season stage interactions, defense trends, elo×spread agreement
8. `compute_sc_matchup_features()` — SuperCoach points allowed to position groups from prior season
9. `compute_referee_features()` — Referee home win bias from rolling history
10. `compute_team_stats_features()` — Prior-season team stats (line breaks, possession %, tackles, etc.)
11. `compute_rolling_match_stats_features()` — Process quality rolling 3/5-game averages
12. `compute_player_form_features()` — Spine player form (run metres, line breaks, try assists) + squad disruption

#### Blend Architecture
```python
# V4 OptBlend (simplest: 35% CatBoost top-60 + 65% odds)
V4_BLEND_WEIGHTS = {"CAT_top50": 0.35}
V4_BLEND_ODDS_WEIGHT = 0.65

# Meta-learner stacking (in predict_round.py)
# Out-of-fold CatBoost predictions + odds probs → LogisticRegression
# Meta-features: model_prob, odds_prob, model×odds, prob_diff, abs_diff, confidence
```

#### Feature Column Specification (194 features)
- V3 base: 131 features (Elo, rolling form, ladder, H2H, venue, odds, momentum, halftime, engineered)
- V4 new: ~63 features (enhanced odds, scoring trends, attendance, kickoff, lineup, player impact, SC matchup, referee, rolling match stats, player form, engineered interactions)

### Architecture

**Data Flow:**
1. Historical matches (RLP scrape) + odds → `link_odds()` fixes home/away
2. Add target: `home_win = (home_score > away_score)`
3. V3 feature pipeline: Elo → form → ladder → venue → odds → engineering
4. V4 feature pipeline: enhanced odds → consistency → attendance → kickoff → lineup → impact → team stats → referee → rolling stats → player form
5. Feature selection: XGBoost identifies top-60 features by importance
6. Train CatBoost on top-60 + sample weights (exponential decay favoring recent years)
7. Get out-of-fold predictions on training data via 5-fold CV
8. Meta-learner (LR): learns optimal blend of model + odds + interaction features
9. Score upcoming matches: CatBoost(top-60) + meta-learned blend → home_win_prob
10. Lineup check: fetch NRL.com teamlists, diff vs baseline, apply player impact adjustments

**Key Files & Ownership:**
- `pipelines/v3.py` — V3 base features (read-only for this task)
- `pipelines/v4.py` — V4 new features, optimization target
- `predict_round.py` — Blend, train, predict, lineup check, output
- `processing/player_impact.py`, `processing/elo.py` — Data processing (read-only)
- `config/settings.py` — Global constants (read-only)

### Patterns & Conventions

**Feature Function Signature:**
```python
def compute_X_features(matches: pd.DataFrame, [optional_arg: Type]) -> pd.DataFrame
```
Returns modified copy, never mutates input. Functions are composable — each builds on prior.

**Look-Ahead Bias Prevention:**
- Rolling stats computed per-team with `iloc[:i]` (only prior games)
- Lineup changes diffed against baseline captured during Tuesday pipeline (pre-game)
- Player impact uses historical impact scores, not game-day lineups
- Referee features use rolling history up to current match

**Column Naming Conventions:**
- Home/away differential: `X_diff` = `home_X - away_X`
- Rolling window: `{stat}_{window}` (e.g., `win_rate_5`, `pf_std_8`)
- Interaction terms: `X_x_Y` (e.g., `elo_diff_x_late`, `fav_consistency`)
- Per-position: `{side}_{position}_{stat}_{window}` (e.g., `home_spine_run_metres_3`)

**Data Import Patterns:**
- Historical: `PROCESSED_DIR / "matches.parquet"`, `"ladders.parquet"`, `"odds.parquet"`
- Player data: `"player_appearances.parquet"`, `"player_match_stats.parquet"`, `"player_impact.parquet"`
- Team stats: `"team_season_stats.parquet"`, `"match_stats.parquet"`, `"sc_points_allowed.parquet"`
- Officials: `"match_officials.parquet"`

**NaN Handling:**
- Training: drop draws + missing scores, then fill with medians (bool→0, numeric→median)
- Prediction: fill coherently from available odds (spread←odds prob, movement←0, etc.), then median fill

### Start Here

**For the 9 improvements task, prioritize modifying:**

1. **`pipelines/v4.py`** — Add new feature functions following existing 12-function pattern. Each function should:
   - Accept `matches` DataFrame (+ optional data source)
   - Build lookups or per-team statistics
   - Attach columns with no look-ahead bias
   - Print progress with column counts
   - Return augmented DataFrame

2. **`predict_round.py`** — Update `FEATURE_COLS` list to include new feature column names. Also:
   - Call new feature functions in `build_features()` after existing V4 calls
   - Adjust meta-learner training if new interaction features are added
   - Update tipping logic if confidence calibration changes

3. **`config/settings.py`** — May need new constants if new feature requires external parameters

**Why This Order:**
- V4 feature functions are independent and composable — each is a vertical slice
- Features appear in `FEATURE_COLS` → included in training automatically
- No changes needed to data loading (all sources already defined)
- Meta-learner retrains automatically on new feature set
- Blend weights may need tuning after feature addition, but grid search is automated

**Acceptance Criteria for New Features:**
- ✅ No look-ahead bias (rolling/prior-data only)
- ✅ Handles missing data gracefully (NaN → median fill or coherent defaults)
- ✅ Coverage metric logged (% non-NaN values)
- ✅ Column names added to `FEATURE_COLS`
- ✅ Type: numeric or bool (tree models can handle; auto-scaled for LR)
- ✅ Feature count fits training memory (target: <250 features)