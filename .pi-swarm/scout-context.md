

### scout-1
I'm a read-only scout preparing context for your 2 builders. Based on my codebase exploration, here's the comprehensive **scout context** you'll hand to the builders:

---

## Scout Context: Add Player-Level Form Features to NRL Prediction Model

### Files Retrieved

1. **scraping/nrl_match_stats.py** (lines 1-250)  
   Current scraper already fetches raw match-centre JSON with player data at `data['stats']['players']` (~58 fields/player), but only extracts team-level stats. The `parse_match_stats()` function processes team stats from `data['stats']['groups']` but ignores player data entirely.

2. **pipelines/v4.py** (lines 441-514, 858-1006)  
   Feature engineering pipeline with `compute_rolling_match_stats_features()` pattern—exactly what builders need to follow. Uses 10 team-level stats × 2 rolling windows (3, 5) = 20 columns per team. Clear no-look-ahead-bias implementation via chronological match logs + explicit prior-match filtering.

3. **processing/build_player_data.py**  
   Already builds `player_appearances.parquet` (85k+ rows): year, round, match_id, team, player_name, position, is_starter, is_spine. Includes jersey→position mapping and SPINE_POSITIONS = {FB, HB, FE, HK}.

4. **processing/player_impact.py**  
   Computes `player_impact.parquet` with weighted_impact scores. Shows Elo-adjustment pattern, though these are all-time scores (not per-match rolling).

5. **config/settings.py**  
   Core constants: PROJECT_ROOT, PROCESSED_DIR, FEATURES_DIR, START_YEAR=2013, END_YEAR=2026, ALL_ROUNDS definition.

### Key Code Patterns

**Player data structure (raw API):**
```python
data['stats']['players'] = {
  'homeTeam': [
    {
      'playerId': 504870,
      'allRuns': 18, 'allRunMetres': 157, 'lineBreaks': 2,
      'tacklesMade': 15, 'missedTackles': 3, 'offloads': 4,
      'passes': 22, 'tackles': 18, 'tries': 0, 'tries': 1,
      'minutesPlayed': 62, 'errors': 1, 'penalties': 2,
      # ... 58 fields total
    },
    # 19 players per team
  ],
  'awayTeam': [...],
  'meta': {...}
}
```

**Existing team-stats extraction (nrl_match_stats.py ~ line 160):**
```python
STAT_COLUMNS = {
    "Possession %": "possession_pct",
    "Completion Rate": "completion_rate",
    "Line Breaks": "line_breaks",
    # 30+ more
}

def parse_match_stats(match_data, year, round_num):
    stats = match_data.get("stats", {})
    groups = stats.get("groups", [])  # TEAM STATS ONLY
    # Ignores: stats.get("players", {})
```

**Rolling features pattern (pipelines/v4.py ~ line 858):**
```python
ROLLING_MATCH_STATS = [
    "completion_rate", "line_breaks", "tackle_breaks", "errors", "missed_tackles",
    "all_run_metres", "possession_pct", "effective_tackle_pct", "post_contact_metres", "offloads",
]
ROLLING_WINDOWS = [3, 5]

def compute_rolling_match_stats_features(matches, match_stats_df):
    # Per-team match log sorted chronologically
    # For each match, lookup last 3 and 5 prior matches
    # Outputs: home_ms_completion_rate_3, away_ms_line_breaks_5, ms_diff_* (total ~60 features)
```

### Architecture

**Current pipeline:**  
NRL.com API → `fetch_match_stats()` → cached JSON → `parse_match_stats()` (TEAM STATS ONLY) → match_stats.parquet (72 cols) → v4.py pipeline → features

**Proposed additions:**
```
1. Extend parse_match_stats() to ALSO extract data['stats']['players']
2. NEW: processing/player_match_stats.py
   - Builds per-player stats per match (raw)
   - Builds rolling form features (3, 5 game rolling averages)
   - Outputs: player_form_features.parquet
3. NEW: compute_player_form_features() in v4.py
   - Aggregates to team level (sum spine, avg efficiency)
   - Joins to matches by (year, round, home_team, away_team)
   - Outputs: home/away_player_form_score_*, *_diff features
```

### Patterns & Conventions

- **File organization:** `scraping/` (fetch), `processing/` (parse → parquet), `pipelines/` (engineer), `config/` (constants)
- **Column naming:** `{side}_{metric}_{window}` (e.g., `home_player_form_5`, `away_tackles_rolling_3`)
- **Joins:** On (year, round, home_team, away_team) OR (year, round, match_id, team, player_name)
- **No look-ahead bias:** Build chronological match logs with match_idx, only reference PRIOR matches
- **Error handling:** Missing data → np.nan, invalid responses → logged warning + skip
- **Imports:** Always `from pathlib import Path`, `PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"`

### Start Here

**Builder 1: Player Stats Extraction & Form Features**  
**Files: scraping/nrl_match_stats.py (extend), processing/player_match_stats.py (NEW)**

1. Extend `parse_match_stats()` to extract `data['stats']['players']`
2. Create `processing/player_match_stats.py` with:
   - `build_player_form_features()` — rolling 3, 5 game stats per player
   - Output: `player_form_features.parquet` with columns: year, round, match_id, team, player_name, player_id, and rolling metrics for 8-10 key stats (allRuns, lineBreaks, tackles, etc.)
3. Test: verify no NaN data leakage, match counts align with appearances.parquet

**Builder 2: Pipeline Integration**  
**Files: pipelines/v4.py (add functions only)**

1. Add `compute_player_form_features(matches, player_form_df)` following the rolling-match-stats pattern exactly
2. Aggregate to team level: sum spine player form (FB, HB, FE, HK), average defensive metrics
3. Add ~12-15 new features to `build_v4_feature_matrix()`: `home/away_player_form_score_3`, `home/away_player_form_score_5`, `player_form_diff_3`, `player_form_diff_5`, etc.
4. Include in walk-forward backtest; verify feature importance

**Dependency:** Builder 1 → Builder 2 (Builder 2 depends on Builder 1's parquet output)

---

**Summary for dispatch:**
- **Complexity:** Moderate (2 files new, 2 files extended, ~300 lines total)
- **Test data:** 2026 season (round 1-5 complete), historical 2013-2025
- **No breaking changes:** Fully backward compatible, gracefully handles missing player_form_features.parquet
- **Timeline:** Builder 1 (player extraction) ~4 hours, Builder 2 (pipeline integration + test) ~3 hours