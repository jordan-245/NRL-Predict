"""
Weekly Data Refresh Pipeline
=============================
Single script for the Monday/Tuesday post-round data refresh.

Steps:
  1. Scrape last completed round from RLP (targeted, not full rebuild)
  2. Update matches.parquet with new results (append, not rebuild)
  3. Update ladders.parquet with current ladder
  4. Update player_appearances.parquet with new lineups
  5. Rebuild player_impact.parquet (recalculate rolling windows)
  6. Invalidate model cache so next prediction run retrains
  7. Optionally record results in tipping tracker

Usage:
    python refresh_week.py                          # auto-detect last round
    python refresh_week.py --round 5 --year 2026    # specific round
    python refresh_week.py --record-tips             # also record tipping results
    python refresh_week.py --full-rebuild            # rebuild everything from scratch
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    END_YEAR, PROCESSED_DIR, ALL_ROUNDS, REGULAR_ROUNDS, FINALS_ROUNDS,
)
from config.team_mappings import standardise_team_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence chatty sub-loggers
for name in ["scraping.rate_limiter", "scraping.rlp_match_parser",
             "scraping.rlp_ladder_parser", "scraping.rlp_scraper"]:
    logging.getLogger(name).setLevel(logging.WARNING)

MODEL_CACHE_DIR = PROJECT_ROOT / "outputs" / "model_cache"


def detect_last_round(year: int) -> int | None:
    """Auto-detect the next round needing results scraped.

    Logic: find the highest round whose fixture dates have passed
    but whose scores are NOT yet in the database. This correctly
    handles catch-up after missed refreshes (e.g. if Round 1 is
    scraped but Round 2 has been played, returns 2 not 1).

    Falls back to start-of-season detection when no scores exist yet.
    """
    matches_path = PROCESSED_DIR / "matches.parquet"
    if not matches_path.exists():
        return None

    df = pd.read_parquet(matches_path)
    year_matches = df[df["year"] == year]

    if year_matches.empty:
        return None

    # Find which rounds already have scores
    played = year_matches.dropna(subset=["home_score"])
    scraped_rounds = set()
    if not played.empty:
        for r in played["round"].unique():
            try:
                scraped_rounds.add(int(r))
            except (ValueError, TypeError):
                pass

    # Find which rounds have fixture dates in the past (games completed)
    now = pd.Timestamp.now()
    completed_rounds = set()
    if "parsed_date" in year_matches.columns:
        # A round is "completed" if ALL its games have kicked off
        for r in year_matches["round"].unique():
            try:
                r_int = int(r)
            except (ValueError, TypeError):
                continue
            round_matches = year_matches[year_matches["round"] == r]
            round_dates = round_matches["parsed_date"].dropna()
            if not round_dates.empty and (round_dates < now).all():
                completed_rounds.add(r_int)

    # The rounds needing scraping = completed but not yet scraped
    unscraped = completed_rounds - scraped_rounds
    if unscraped:
        # Return the LOWEST unscraped round (process in order)
        target = min(unscraped)
        if scraped_rounds:
            print(f"  (catch-up: rounds {sorted(scraped_rounds)} already scraped, "
                  f"round {target} needs scraping)")
        else:
            print(f"  (start-of-season: round {target} has completed but no scores yet)")
        return target

    # Everything is caught up — return None (nothing to scrape)
    if scraped_rounds:
        print(f"  (all completed rounds already scraped: {sorted(scraped_rounds)})")
    return None


def _standardise_team_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply standardise_team_name to columns."""
    for col in cols:
        if col not in df.columns:
            continue
        result = []
        for val in df[col]:
            if pd.isna(val) or str(val).strip() == "":
                result.append(val)
                continue
            try:
                result.append(standardise_team_name(str(val)))
            except KeyError:
                result.append(str(val))
        df[col] = result
    return df


def step1_scrape_round(year: int, round_num: int) -> tuple[list[dict], list[dict]]:
    """Scrape a single round's results and ladders from RLP."""
    from scraping.rlp_scraper import RLPScraper
    from scraping.rlp_match_parser import parse_round_summary
    from scraping.rlp_ladder_parser import parse_round_ladder

    print(f"\n  STEP 1: Scraping Round {round_num} of {year} from RLP...")

    scraper = RLPScraper(use_cache=False, show_progress=False)

    # Scrape round summary
    html = scraper.scrape_round_summary(year, round_num)
    matches = []
    if html:
        matches = parse_round_summary(html, year=year, round_id=round_num)
        print(f"    Parsed {len(matches)} matches")

    # Scrape ladder
    ladder_html = scraper.scrape_round_ladder(year, round_num)
    ladders = []
    if ladder_html:
        ladders = parse_round_ladder(ladder_html, year, round_num)
        print(f"    Parsed {len(ladders)} ladder rows")

    return matches, ladders


def step_scrape_match_stats(year: int, round_num: int):
    """Scrape per-game team stats from NRL.com API for the just-completed round.

    Fetches completion rate, line breaks, tackle breaks, errors, etc. for
    all matches in the round and appends them to match_stats.parquet.
    Idempotent: removes any existing rows for this year+round before saving.
    """
    print(f"\n  STEP 1b: Scraping match stats for Round {round_num} of {year}...")

    try:
        from scraping.nrl_match_stats import fetch_round_match_stats
    except ImportError as e:
        print(f"    SKIPPED: could not import fetch_round_match_stats — {e}")
        return

    try:
        new_stats = fetch_round_match_stats(year, round_num)
    except Exception as e:
        print(f"    ERROR fetching match stats: {e}")
        return

    if not new_stats:
        print(f"    No stats returned for Round {round_num} — skipping")
        return

    new_df = pd.DataFrame(new_stats)
    if new_df.empty:
        print(f"    Empty stats DataFrame — skipping")
        return

    print(f"    Fetched {len(new_df)} match stat rows")

    stats_path = PROCESSED_DIR / "match_stats.parquet"
    if stats_path.exists():
        existing = pd.read_parquet(stats_path)
        # Remove existing rows for this year+round (idempotent)
        mask = ~((existing["year"] == year) & (existing["round"].astype(str) == str(round_num)))
        existing = existing[mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(stats_path, index=False, engine="pyarrow")
    print(f"    Saved match_stats.parquet — {len(combined)} total rows "
          f"({len(new_df)} new for Round {round_num})")


def step_scrape_player_stats(year: int, round_num: int):
    """Scrape per-player match stats from NRL.com match centre API.

    Fetches individual player stats (run metres, tackles, line breaks, etc.)
    for all players in all matches of the round and appends them to
    player_match_stats.parquet.
    Idempotent: removes any existing rows for this year+round before saving.
    """
    print(f"\n  STEP 1c: Scraping player stats for Round {round_num} of {year}...")

    try:
        from scraping.nrl_player_stats import fetch_round_player_stats
    except ImportError as e:
        print(f"    SKIPPED: could not import fetch_round_player_stats — {e}")
        return

    try:
        new_stats = fetch_round_player_stats(year, round_num)
    except Exception as e:
        print(f"    ERROR fetching player stats: {e}")
        return

    if not new_stats:
        print(f"    No player stats returned for Round {round_num} — skipping")
        return

    new_df = pd.DataFrame(new_stats)
    if new_df.empty:
        print(f"    Empty player stats DataFrame — skipping")
        return

    print(f"    Fetched {len(new_df)} player stat rows")

    stats_path = PROCESSED_DIR / "player_match_stats.parquet"
    if stats_path.exists():
        existing = pd.read_parquet(stats_path)
        # Remove existing rows for this year+round (idempotent)
        mask = ~((existing["year"] == year) & (existing["round"].astype(str) == str(round_num)))
        existing = existing[mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    # Deduplicate on (year, round, match_id, player_id)
    dedup_cols = ["year", "round", "match_id", "player_id"]
    if all(c in combined.columns for c in dedup_cols):
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")

    combined.to_parquet(stats_path, index=False, engine="pyarrow")
    print(f"    Saved player_match_stats.parquet — {len(combined)} total rows "
          f"({len(new_df)} new for Round {round_num})")


def step2_update_matches(new_matches: list[dict], year: int, round_num: int):
    """Append new match results to matches.parquet."""
    print(f"\n  STEP 2: Updating matches.parquet...")

    matches_path = PROCESSED_DIR / "matches.parquet"

    if not new_matches:
        print("    No new matches to add")
        return

    new_df = pd.DataFrame(new_matches)

    # Drop complex columns (lists/dicts) for Parquet compatibility
    drop_cols = []
    for col in new_df.columns:
        sample = new_df[col].dropna().head(5)
        if any(isinstance(v, (list, dict)) for v in sample):
            drop_cols.append(col)
    if drop_cols:
        new_df = new_df.drop(columns=drop_cols)

    # Standardise teams
    _standardise_team_columns(new_df, ["home_team", "away_team"])

    # Fix home/away using venue data
    from processing.venue_home_fix import fix_home_away
    new_df = fix_home_away(new_df)

    # Ensure round is string
    new_df["round"] = new_df["round"].astype(str)

    if matches_path.exists():
        existing = pd.read_parquet(matches_path)
        # Remove existing data for this round (idempotent update)
        mask = ~((existing["year"] == year) & (existing["round"] == str(round_num)))
        existing = existing[mask]

        # Align datetime resolution (parquet stores us, scraper returns ns)
        for col in existing.columns:
            if pd.api.types.is_datetime64_any_dtype(existing[col]) and col in new_df.columns:
                new_df[col] = new_df[col].astype(existing[col].dtype)

        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(matches_path, index=False, engine="pyarrow")
    print(f"    Updated: {len(combined)} total matches")


def step3_update_ladders(new_ladders: list[dict], year: int, round_num: int):
    """Append new ladder data to ladders.parquet."""
    print(f"\n  STEP 3: Updating ladders.parquet...")

    ladders_path = PROCESSED_DIR / "ladders.parquet"

    if not new_ladders:
        print("    No new ladder data")
        return

    new_df = pd.DataFrame(new_ladders)
    _standardise_team_columns(new_df, ["team"])
    new_df["round"] = new_df["round"].astype(str)

    if ladders_path.exists():
        existing = pd.read_parquet(ladders_path)
        mask = ~((existing["year"] == year) & (existing["round"] == str(round_num)))
        existing = existing[mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(ladders_path, index=False, engine="pyarrow")
    print(f"    Updated: {len(combined)} total ladder rows")


def step4_update_player_appearances(year: int, round_num: int):
    """Update player appearances with the new round's lineups."""
    print(f"\n  STEP 4: Updating player_appearances.parquet...")

    from processing.build_player_data import append_round, OUTPUT_PATH

    try:
        df = append_round(year, round_num)
        if not df.empty:
            df.to_parquet(OUTPUT_PATH, index=False)
            print(f"    Updated: {len(df)} total appearances")
        else:
            print("    No lineup data found for this round")
    except FileNotFoundError as e:
        print(f"    SKIPPED: {e}")
    except Exception as e:
        print(f"    ERROR: {e}")


def step5_rebuild_player_impact():
    """Rebuild player impact scores (fast — just recalculates rolling windows)."""
    print(f"\n  STEP 5: Rebuilding player impact scores...")

    from processing.player_impact import load_data, build_team_match_log, compute_impact_scores, OUTPUT_PATH

    try:
        appearances, matches = load_data()
        team_log = build_team_match_log(matches)
        impact_df = compute_impact_scores(appearances, team_log)

        if not impact_df.empty:
            impact_df.to_parquet(OUTPUT_PATH, index=False)
            print(f"    Saved: {len(impact_df)} player impact scores")
        else:
            print("    No impact scores computed (insufficient data)")
    except FileNotFoundError as e:
        print(f"    SKIPPED: {e}")
    except Exception as e:
        print(f"    ERROR: {e}")


def step6_invalidate_cache(year: int):
    """Remove model caches so next prediction retrains with fresh data."""
    print(f"\n  STEP 6: Invalidating model cache...")

    if not MODEL_CACHE_DIR.exists():
        print("    No cache directory found")
        return

    removed = 0
    for cache_file in MODEL_CACHE_DIR.glob(f"*_{year}.joblib"):
        cache_file.unlink()
        removed += 1

    print(f"    Removed {removed} cache files")


def step7_record_tips(year: int, round_num: int):
    """Record tipping results for the completed round."""
    print(f"\n  STEP 7: Recording tipping results...")

    try:
        from tipping_tracker import record_round
        record_round(round_num, year, auto=True)
    except ImportError as e:
        print(f"    tipping_tracker.py not available ({e}), skipping")
    except Exception as e:
        print(f"    ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Weekly NRL data refresh")
    parser.add_argument("--round", type=int, default=None,
                        help="Round to refresh (auto-detected if omitted)")
    parser.add_argument("--year", type=int, default=END_YEAR,
                        help=f"Season year (default: {END_YEAR})")
    parser.add_argument("--record-tips", action="store_true",
                        help="Also record tipping results")
    parser.add_argument("--full-rebuild", action="store_true",
                        help="Full rebuild of player data (not incremental)")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip RLP scraping (just rebuild derived data)")
    args = parser.parse_args()

    t_start = time.time()
    year = args.year

    print("=" * 60)
    print(f"  NRL Weekly Data Refresh — {year}")
    print("=" * 60)

    # Auto-detect round(s)
    if args.round is not None:
        rounds_to_process = [args.round]
        print(f"\n  Target round: {args.round}")
    else:
        # Detect ALL rounds needing scraping (catch-up mode)
        rounds_to_process = []
        while True:
            r = detect_last_round(year)
            if r is None:
                break
            if r in rounds_to_process:
                break  # safety: prevent infinite loop
            rounds_to_process.append(r)
            # Temporarily mark this round as "will be processed" by
            # breaking — we'll loop in the processing section below
            break

        if not rounds_to_process:
            print("\n  No rounds need scraping — all caught up!")
            print("  Use --round N to force a specific round.")
            sys.exit(0)
        print(f"\n  Auto-detected round(s) to process: {rounds_to_process}")

    def _process_round(round_num: int):
        """Process a single round (scrape + update)."""
        print(f"\n  {'─' * 50}")
        print(f"  Processing Round {round_num}")
        print(f"  {'─' * 50}")

        if args.full_rebuild:
            print("\n  FULL REBUILD MODE — rebuilding all player data from scratch")
            from processing.build_player_data import build_full, OUTPUT_PATH as APP_PATH
            df = build_full()
            if not df.empty:
                df.to_parquet(APP_PATH, index=False)
            step5_rebuild_player_impact()
            step6_invalidate_cache(year)
        else:
            # Incremental update
            if not args.skip_scrape:
                new_matches, new_ladders = step1_scrape_round(year, round_num)
                step_scrape_match_stats(year, round_num)
                step_scrape_player_stats(year, round_num)
                step2_update_matches(new_matches, year, round_num)
                step3_update_ladders(new_ladders, year, round_num)
            else:
                print("\n  Skipping RLP scrape (--skip-scrape)")

            step4_update_player_appearances(year, round_num)
            step5_rebuild_player_impact()
            step6_invalidate_cache(year)

        if args.record_tips:
            step7_record_tips(year, round_num)

    # Process first detected round
    _process_round(rounds_to_process[0])

    # Catch-up loop: check if more rounds need scraping
    if args.round is None:
        while True:
            next_round = detect_last_round(year)
            if next_round is None:
                break
            print(f"\n  ⚡ Catch-up: Round {next_round} also needs scraping")
            _process_round(next_round)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Refresh completed in {elapsed:.1f}s")
    print(f"  Next: python predict_round.py --auto")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
