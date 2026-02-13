# NRL Late Team Selection Changes — Edge Detection System

## Context

Our honest evaluation experiment confirmed that no ML model or ensemble beats bookmaker odds (LL=0.5999) in walk-forward testing. The betting market is too efficient for a general prediction edge using public features.

**The pivot**: Instead of trying to beat the market on average, exploit a specific information asymmetry — late team selection changes. Odds are set based on Tuesday's 21-man squad. When the final 17 is confirmed ~1 hour before kickoff, key player changes (especially spine positions) can shift true win probability by 5-10%, but bookmakers don't always adjust fully or instantly.

**Goal**: Build a player impact model that detects when a late team change creates a probability gap between our model's adjustment and the market's adjustment, then bet selectively on those 20-30 games per season.

**Critical prerequisite**: Before building anything, we must confirm two things:
1. The pattern actually exists (Phase 0 — manual check)
2. We can get multi-timestamp odds AND actually place bets in the pre-game window (Phase 0.5)

---

## What We Already Have

### Existing Data (2,560 matches, 2013-2025)
- **89,006 player-match records** in `data/processed/player_match_stats.parquet` — 72 columns including `fantasyPointsTotal`, `allRunMetres`, `tacklesMade`, position, jersey number
- **2,560 NRL.com JSON files** in `data/raw/nrl_match_stats/nrl_api/` — each contains the **final playing 17-18** (post-match lineups) with `firstName`, `lastName`, `position`, `number`, `isOnField`
- **Player quality features** already computed: rolling team fantasy, spine fantasy, forward stats
- **Full match outcomes + odds data** for backtesting
- **NRL.com API pattern**: `https://www.nrl.com/draw/nrl-premiership/{year}/{round_slug}/{home_slug}-v-{away_slug}/data`

### Key Insight from Data
- **Fantasy points by position** confirm impact asymmetry:
  - Halfback (45.0 avg), Lock (44.6), Hooker (44.4), 2nd Row (42.5) — high impact
  - Interchange (25.2), Reserve (1.2), Replacement (0.1) — low impact
  - Top player (Nathan Cleary) averages 65.3 fantasy — losing him is ~20 points above a replacement halfback
- Post-match JSONs have 17-18 players (final team), not the 21-man Tuesday squad
- **Position impact should be LEARNED from data, not hardcoded** — Phase 2 will discover the actual weights

### What We Don't Have Yet
- **Tuesday team lists** (the 21-man announced squad) — not in our data
- **Late Mail changes** (who was dropped/added between Tuesday and game day)
- **Multi-timestamp odds** (opening vs closing, or pre/post team change) — CRITICAL gap
- **Game-day final team** separate from post-match data

---

## Data Sources for Team List Changes

### Source 1: NRL.com Team List Articles (Tuesday squads)
- **URL pattern**: `https://www.nrl.com/news/{YYYY}/{MM}/{DD}/nrl-team-lists-round-{N}/`
- **Format**: HTML article with 21-man squads (17 starters + 4 reserves) per match
- **Structure**: Numbered players with positions (Backs, Forwards, Interchange, Reserve)
- **Coverage**: 2024-2026 confirmed, likely 2020+ in archives
- **Published**: Tuesdays at 4pm AEDT

### Source 2: NRL.com Late Mail Articles (game-day changes)
- **URL pattern**: `https://www.nrl.com/news/{YYYY}/{MM}/{DD}/nrl-late-mail-round-{N}/`
- **Format**: Narrative prose describing changes per team (not structured IN/OUT lists)
- **Coverage**: Same period as team list articles
- **Published**: 24 hours before kickoff, then game day updates

### Source 3: NRL.com Match Data API (final playing team — already cached)
- **URL pattern**: `https://www.nrl.com/draw/nrl-premiership/{year}/{round_slug}/{home_slug}-v-{away_slug}/data`
- **Format**: JSON with exact 17-18 players who played
- **Coverage**: 2013-2025 (2,560 files cached)
- **This is the FINAL team** — but only available after the game starts

### Source 4: League Unlimited (structured change tracking)
- **URL pattern**: `https://leagueunlimited.com/news/leagueunlimited-nrl-teams-{YYYY}-round-{N}/`
- **Format**: Shows 22-man squad → 19-man update → final 17+1 with explicit change descriptions
- **Coverage**: Historical archives available

### Recommended Approach
Use **Source 1 (Tuesday article)** + **Source 3 (final playing team from cached JSON)** to derive changes by diffing the two lists. This avoids parsing narrative Late Mail text.

**Name matching warning**: This is the hardest part. Tuesday articles say "Nathan Cleary" while JSON may have "Nathan J Cleary" or different formatting. The fuzzy matching layer must be built early and validated thoroughly — if the diff is wrong, everything downstream is wrong. Budget significant time here. Consider building a manual override mapping file for known mismatches.

---

## Implementation Plan

### Phase 0: Manual Quick Validation (~2 hours)

**Before writing any code**, manually check 10-15 high-profile late changes from 2024-2025:
- Find well-known cases: Nathan Cleary missing games, starting halfbacks scratched on game day, key fullback late outs
- For each: what were the odds before the change? Did the result go against those odds?
- This tells us if the pattern even exists before investing in automation
- Sources: NRL.com Late Mail articles + memory of big 2024-2025 team news

**Go/no-go**: If fewer than 60% of big late changes align with the expected direction, the edge may not exist.

### Phase 0.5: Verify Market Lag + Betting Feasibility (~1 day)

**This phase is critical and was missing from the original plan.**

**Step 0.5a: Multi-timestamp odds data**
- Investigate whether our existing odds data (AusSportsBetting Excel) has opening AND closing odds
- Research available sources for historical odds snapshots:
  - Odds comparison sites (OddsPortal, Oddschecker) — may have historical line movement
  - Betfair exchange — has full price history, may have API access
  - BetsAPI (`betsapi.com/l/298/NRL`) — may provide odds history
  - Pinnacle/Bet365 — some provide historical data via API
- **What we need**: for each match, odds at (a) Tuesday after team announcement, (b) 1 hour before kickoff after final team, (c) closing odds
- If closing odds fully reflect the team change, there's no window to exploit and the whole strategy fails

**Step 0.5b: Betting window feasibility**
- Verify you can actually place bets in the 30-60 minute window between final team confirmation and kickoff
- Check: do retail bookmakers (Sportsbet, TAB, Ladbrokes, Bet365) suspend or limit markets during that window?
- Check: is there sufficient liquidity on Betfair exchange pre-game?
- Check: are there account limiting risks for consistently betting at the last minute?
- **This is a practical constraint that kills the strategy even if the edge is real**

**Go/no-go**: If closing odds fully price in team changes, OR if you can't place bets in the window, stop here.

### Phase 1: Historical Team List Scraper (~2-3 days)

**Script**: `scrape_team_lists.py`

**Step 1a: Build the name matching layer FIRST**
- Before scraping, build and validate the fuzzy name matching system
- Create a canonical player name registry from `player_match_stats.parquet` (89k records, unique player names per team per year)
- Implement matching: exact → normalized (strip suffixes/prefixes, lowercase) → fuzzy (Levenshtein, token sort ratio) → manual override file
- Create `config/player_name_overrides.json` for known mismatches
- **Validation**: test against a sample of 50 manually verified name pairs
- This is the foundation — everything downstream depends on it

**Step 1b: Discover team list article URLs**
- Scrape `https://www.nrl.com/news/topic/team-lists/` paginated listing
- Extract all article URLs matching pattern `nrl-team-lists-round-*`
- Target: 2020-2025 (6 seasons × ~30 rounds = ~180 articles)

**Step 1c: Parse Tuesday team list articles**
- For each article, extract the 21-man squad per match:
  - Parse HTML for player entries: jersey number, position, full name
  - Group by match (home team vs away team)
  - Identify the 17 named starters + 4 reserves
- Output: `data/processed/tuesday_squads.parquet`
  - Columns: `year, round, home_team, away_team, side, jersey_number, position, player_name, canonical_name, is_reserve`

**Step 1d: Build the "final team" reference from cached JSON**
- Already have 2,560 post-match JSONs with the playing 17-18
- Extract: `year, round_slug, team, player_name, canonical_name, position, number`
- Output: `data/processed/final_teams.parquet`

**Step 1e: Diff Tuesday vs Final to get changes**
- Match by team + round using canonical names
- Identify: players in Tuesday squad but not in final team (DROPPED)
- Identify: players in final team but not in Tuesday squad (ADDED)
- Flag unmatched names for manual review
- Output: `data/processed/team_changes.parquet`
  - Columns: `year, round, team, match_id, change_type (IN/OUT), player_name, canonical_name, position, jersey_number`

**Key files to create**:
- `scrape_team_lists.py` (new) — scrape + parse NRL.com team list articles
- `extract_team_changes.py` (new) — diff Tuesday vs final, produce changes parquet
- `config/player_name_overrides.json` (new) — manual name mapping fixes
- Reuse: `scraping/rate_limiter.py`, `config/team_mappings.py`, `config/settings.py`

### Phase 2: Player Impact Model (~2 days)

**Script**: `build_player_impact_model.py`

**Step 2a: Build player value ratings**
- Use `player_match_stats.parquet` (89k records) to compute per-player rolling metrics:
  - Rolling 5-game and 8-game average fantasy points
  - Rolling average for key stats: `allRunMetres`, `tacklesMade`, `lineBreaks`, `tryAssists`, `kickMetres`
  - Career games played (experience proxy)
  - Position-specific impact metrics (e.g., `kickMetres` for halfbacks, `tacklesMade` for forwards)
- Compute a single **Player Impact Score (PIS)** per player per match window
- Output: `data/processed/player_ratings.parquet`

**Step 2b: Compute replacement value by position**
- For each team + position, calculate the typical quality of a replacement:
  - Average fantasy of the next player on the team's depth chart at that position
  - League-average for the position as fallback
- This gives us "Value Over Replacement" (VOR) — how much the team loses when player X is dropped

**Step 2c: Learn position impact weights from data**
- **Do NOT hardcode position weights** (e.g., "halfback = 3x") — learn them from historical data
- For each late change in `team_changes.parquet`:
  - Look up the dropped player's PIS and the replacement's PIS
  - Calculate the quality delta
  - Group by position of the changed player
  - Measure actual outcome deviation from odds expectation per position group
- Let the data reveal: hooker changes might matter more than halfback changes, or prop changes might be more impactful in certain contexts than assumed
- Output: learned position weights + **Team Change Impact Score** formula
- Output: `data/processed/change_impact_analysis.parquet`

### Phase 3: Edge Detection & Backtesting (~2 days)

**Script**: `detect_late_change_edge.py`

**Step 3a: Historical backtesting**
- For each match with late changes:
  - Calculate our probability adjustment based on the Team Change Impact Score
  - Compare to: (a) opening odds (Tuesday), (b) closing odds (if multi-timestamp available)
  - Measure whether our adjustment better predicted the outcome than the market's adjustment
- Walk-forward evaluation: train on 2020-2022, test on 2023-2025

**Step 3b: Optimize betting triggers from data**
- **Do NOT use an arbitrary 3% threshold** — optimize from the backtest
- Test a range of thresholds (1% to 10%) and measure ROI at each
- Test whether threshold should vary by position of the changed player or by magnitude of change
- Measure at each threshold: hit rate, ROI, sample size, statistical significance
- The optimal threshold is where ROI is maximized AND sample size is sufficient (>30 bets per year for statistical power)

**Step 3c: Build the selective betting dashboard**
- Show: which matches this week have significant late changes
- For each: our probability shift, market's probability shift, recommended bet (if any)
- Confidence level based on the magnitude of the change and historical accuracy

### Phase 4: Production Game-Day Workflow (~1 day)

**Script**: `gameday_edge_detector.py`

**Workflow**:
1. **Tuesday**: Scrape team list article → store 21-man squads
2. **Run baseline**: Record Tuesday odds as baseline
3. **Game day (1 hour before kickoff)**: Scrape final team from NRL.com OR manual input of changes
4. **Diff**: Compare final team to Tuesday squad
5. **Score change**: Apply Player Impact Model to compute probability adjustment
6. **Compare to market**: Check current odds, calculate if our adjustment > market adjustment by threshold → flag as betting opportunity
7. **Output**: Alert with match, change details, our adjusted probability, market probability, recommended action

---

## Validation Strategy

### Historical Backtest (the honest test)
1. For 2023-2025 matches with late changes, compute our predicted edge
2. Simulate selective betting with optimized threshold (not arbitrary 3%)
3. Measure:
   - **Hit rate**: do we pick winners >50% on triggered bets?
   - **ROI**: with $100 flat stakes on triggered bets, what's the return?
   - **Calibration**: is our probability shift accurate?
   - **Statistical significance**: is the sample size large enough? (need 50-100+ triggered bets)

### Sanity checks
- Verify that spine late outs have historically worse outcomes for the affected team than bench swaps
- **Verify the market doesn't fully price in game-day changes** — this is the entire thesis. Compare opening vs closing odds on matches with big late changes
- Check for survivorship bias: are we only noticing the changes that mattered?

### Key risk the plan must address
Even if the edge exists historically, the window is tiny (30-60 minutes). Retail bookmakers may suspend or limit markets during that window. You need to verify that you can actually place bets with sufficient liquidity in that timeframe. **Check this in Phase 0.5 before building anything.**

---

## File Structure

```
NRL-Predict/
├── scrape_team_lists.py          # Phase 1: scrape Tuesday squad articles
├── extract_team_changes.py       # Phase 1: diff Tuesday vs final teams
├── build_player_impact_model.py  # Phase 2: player ratings + change scoring
├── detect_late_change_edge.py    # Phase 3: historical backtest + edge detection
├── gameday_edge_detector.py      # Phase 4: production game-day workflow
├── config/
│   └── player_name_overrides.json # Manual name mapping fixes
├── data/
│   ├── raw/
│   │   └── nrl_team_lists/       # Cached team list article HTML
│   └── processed/
│       ├── tuesday_squads.parquet
│       ├── final_teams.parquet
│       ├── team_changes.parquet
│       ├── player_ratings.parquet
│       └── change_impact_analysis.parquet
```

---

## Execution Order

| Phase | What | Time | Gate |
|-------|------|------|------|
| **0** | Manual check: 10-15 big late changes from 2024-2025 | 2 hours | >60% align with expected direction |
| **0.5** | Verify multi-timestamp odds exist + betting window is open | 1 day | Can get opening/closing odds AND can place bets pre-game |
| **1** | Name matching layer + team list scraper + change extraction | 2-3 days | >90% name match rate on validated sample |
| **2** | Player impact model (learned weights, not hardcoded) | 2 days | Position weights make intuitive sense |
| **3** | Edge detection backtest with optimized thresholds | 2 days | Positive ROI on walk-forward test with sufficient sample |
| **4** | Production game-day workflow | 1 day | End-to-end test on upcoming round |

**Each phase has a go/no-go gate.** If Phase 0 or 0.5 fails, we stop and save a week of effort.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Name matching failures** | HIGH | Build fuzzy matching early, validate on 50+ pairs, maintain manual override file |
| **Market fully prices changes** | CRITICAL | Phase 0.5 checks opening vs closing odds — if gap is <1%, no edge exists |
| **Can't bet in the window** | CRITICAL | Phase 0.5 verifies bookmaker availability, exchange liquidity, account limits |
| **Small sample size** | MEDIUM | Need 3+ years of data, accept wider confidence intervals, don't overfit thresholds |
| **Scraping fragility** | LOW | Cache all HTML, use flexible parsing, NRL.com article format is fairly stable |
| **Hardcoded assumptions** | MEDIUM | Learn position weights and thresholds from data, not intuition |
