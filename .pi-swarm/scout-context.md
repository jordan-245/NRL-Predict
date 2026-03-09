

### scout-1
Perfect! Now I have comprehensive codebase context. Let me create a detailed scout report:

---

## SCOUT REPORT: Rolling Per-Game Match Stats Features

### Files Retrieved

1. **`scraping/nrl_match_stats.py`** (lines 1-100) — Defines 30+ team-level stats mapping from NRL.com API (possession_pct, line_breaks, tackle_breaks, completion_rate, etc.). Saves to `data/processed/match_stats.parquet`.

2. **`data/processed/match_stats.parquet`** (exists but sparse) — 568 rows (2020-2022) with columns for home_* and away_* stats. Schema present but data mostly NULL (only 2015 has actual values in cache).

3. **`pipelines/v3.py`** (lines 385-485) — `compute_rolling_form_features()` function implements the rolling window pattern we need to emulate: builds team match log, computes windows [3,5,8], creates lookup dict, attaches to DataFrame.

4. **`pipelines/v4.py`** (lines 1-100, 1676-1720) — Main pipeline orchestration. V4 already computes 13 feature groups (elo, odds, form, venue, etc.) but does NOT load match_stats yet. Functions are called sequentially in `main()`.

5. **`config/feature_config.yaml`** (lines 1-250) — Defines rolling windows [3,5,8,10], EWMA half-lives [3,5,8], and metrics to aggregate (win_rate, points_scored, etc.). `advanced_stats` feature group exists in v4+ but is not yet implemented.

6. **`scripts/backtest_team_stats.py`** (lines 1-100) — Shows pattern for testing new stats features via walk-forward backtest. Uses season-level prior stats merged onto matches.

### Key Code

**Pattern from v3.py** (rolling form implementation):
```python
# 1. Build team match log (one row per team per match)
home_log = pd.DataFrame({
    "match_idx": range(len(df)), "team": df["home_team"],
    "points_for": df["home_score"], "points_against": df["away_score"],
    "date": df["date"],
})
log = pd.concat([home_log, away_log], ignore_index=True)
log = log.sort_values(["team", "date", "match_idx"]).reset_index(drop=True)

# 2. For each team/match, compute rolling stats on PRIOR games
lookup = {}
for team in log["team"].unique():
    t_log = log[log["team"] == team].reset_index(drop=True)
    for i, row in t_log.iterrows():
        midx = int(row["match_idx"])
        key = (team, midx)
        prior = t_log.iloc[:i]
        
        for w in windows:  # [3, 5, 8]
            pw = prior.tail(w)
            lookup[key][f"win_rate_{w}"] = pw["win"].mean()

# 3. Attach to DataFrame by (team, match_idx) lookup
for side, team_col in [("home", "home_team"), ("away", "away_team")]:
    for w in windows:
        col_name = f"{side}_win_rate_{w}"
        df[col_name] = [lookup.get((df.iloc[i][team_col], i), {}).get(f"win_rate_{w}", np.nan) for i in range(len(df))]
```

**Match stats schema** (from STAT_COLUMNS in nrl_match_stats.py):
```python
STAT_COLUMNS = {
    "Possession %": "possession_pct",
    "Completion Rate": "completion_rate",
    "Line Breaks": "line_breaks",
    "Tackle Breaks": "tackle_breaks",
    "All Run Metres": "all_run_metres",
    "Post Contact Metres": "post_contact_metres",
    "Errors": "errors",
    "Penalties Conceded": "penalties_conceded",
    "Effective Tackle %": "effective_tackle_pct",
    "Missed Tackles": "missed_tackles",
    # ... 20+ more stats
}
```

### Architecture

```
match_stats.parquet (568 matches, 2020-2022)
    ↓
[NEW] processing/rolling_match_stats.py
    ├─ load_match_stats() → reads parquet
    ├─ compute_rolling_match_stats() → implements rolling window logic
    │   (per v3 pattern: build log, compute windows [3,5,8,10], lookup)
    └─ generates: home_possession_pct_3, home_possession_pct_5, etc.
                  away_possession_pct_3, away_line_breaks_5, etc.
    ↓
pipelines/v4.py main()
    ├─ [MODIFIED] Add call to compute_rolling_match_stats()
    ├─ [UPDATED] build_v4_feature_matrix() to include new cols
    └─ Feed into walk-forward backtest
```

**Data Flow:**
- match_stats.parquet has year, round, home_team, away_team, home_possession_pct, away_possession_pct, etc.
- matches.parquet has same join keys (year, round, home_team, away_team)
- Join on (year, round, home_team, away_team) or sort by date + match_idx
- Build per-team time series, compute rolling windows
- Return DataFrame with new feature columns to attach back to matches

### Patterns & Conventions

- **Naming**: `{side}_{stat}_{window}` where side ∈ {home, away}, stat ∈ {possession_pct, line_breaks, ...}, window ∈ {3, 5, 8, 10}
- **Null handling**: Use `np.nan` for games where insufficient prior history (first 3-5 games have no/partial window data)
- **Window sizes**: From `config/feature_config.yaml` → [3, 5, 8, 10] games
- **Differentials**: After computing per-team features, create diff cols: `home_possession_pct_5 - away_possession_pct_5`
- **Column selection**: Match stats are already home_* / away_* separated in parquet; no need to unpivot
- **Data coverage**: 2015 has real data, 2020-2022 have empty columns. Code must handle gracefully (pass NaN through)
- **Testing pattern**: Use backtest_team_stats.py as reference for walk-forward validation

### Start Here

**File Ownership for a Swarm (if multiple builders):**
| Builder | Owns |
|---------|------|
| Builder 1 | `processing/rolling_match_stats.py` (new module) |
| Builder 2 | `pipelines/v4.py` (integrate function + update main) |
| Builder 3 | `tests/test_rolling_match_stats.py` (new tests) + validation |

**Builder 1 should start with:**
1. Create `processing/rolling_match_stats.py` with functions:
   - `load_match_stats(years=None)` → reads parquet, optionally filters by year
   - `compute_rolling_match_stats_features(matches, match_stats, windows=[3,5,8,10])` → implements rolling logic per v3 pattern
   - Returns matches DataFrame with new columns appended

2. Key implementation points:
   - Reuse the v3 rolling form logic (build log, per-team iterator, lookup dict)
   - Apply to each stat in match_stats columns (possession_pct, line_breaks, etc.)
   - Handle sparse/NULL data: if stat col is all NaN, skip gracefully
   - Ensure no look-ahead bias: only use prior games in window
   - Sort by (team, date, match_idx) for chronological order

**Builder 2 should integrate in v4.py:**
1. Import and call after v3 base features:
   ```python
   from processing.rolling_match_stats import load_match_stats, compute_rolling_match_stats_features
   ...
   match_stats = load_match_stats()
   if not match_stats.empty:
       matches = compute_rolling_match_stats_features(matches, match_stats)
   ```
2. Update `build_v4_feature_matrix()` to include new rolling stat columns
3. Document which stats were included in feature summary

**Builder 3 creates tests:**
1. Unit tests for rolling computation (manual dataframe, verify window math)
2. Integration test: ensure no NaN bleed where data exists
3. Walk-forward backtest: compare accuracy with/without rolling match stats

---

This structure allows independent parallel work with clear boundaries and no file conflicts.