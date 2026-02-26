"""
Custom NRL data scraping and processing pipeline.

Scrapes round summary and ladder data from Rugby League Project for all
seasons 2013-2025, loads odds from the AusSportsBetting Excel file,
standardises team names, and saves everything as Parquet files.

Usage:
    python run_scrape.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config.settings import (
    END_YEAR,
    FINALS_ROUNDS,
    PROCESSED_DIR,
    REGULAR_ROUNDS,
    START_YEAR,
)
from config.team_mappings import standardise_team_name
from scraping.rlp_scraper import RLPScraper
from scraping.rlp_match_parser import parse_round_summary
from scraping.rlp_ladder_parser import parse_round_ladder
from scraping.odds_loader import load_odds

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence overly verbose loggers for cleaner output
logging.getLogger("scraping.rate_limiter").setLevel(logging.WARNING)
logging.getLogger("scraping.rlp_match_parser").setLevel(logging.WARNING)
logging.getLogger("scraping.rlp_ladder_parser").setLevel(logging.WARNING)
logging.getLogger("scraping.rlp_scraper").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Team name standardisation helper
# ---------------------------------------------------------------------------

def standardise_team_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply standardise_team_name to the specified columns, in-place.

    Unknown team names are logged and left unchanged.
    """
    for col in cols:
        if col not in df.columns:
            continue
        standardised = []
        for val in df[col]:
            if pd.isna(val) or str(val).strip() == "":
                standardised.append(val)
                continue
            try:
                standardised.append(standardise_team_name(str(val)))
            except KeyError:
                logger.warning("Unknown team name in column '%s': '%s'", col, val)
                standardised.append(str(val))
        df[col] = standardised
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    print("=" * 70)
    print("  NRL Data Scraping Pipeline")
    print(f"  Seasons: {START_YEAR} - {END_YEAR}")
    print(f"  Regular rounds: 1-{REGULAR_ROUNDS.stop - 1}")
    print(f"  Finals rounds: {FINALS_ROUNDS}")
    print(f"  Output directory: {PROCESSED_DIR}")
    print("=" * 70)

    # Create the scraper with caching enabled
    scraper = RLPScraper(use_cache=True, show_progress=True)

    all_matches: list[dict] = []
    all_ladders: list[dict] = []

    # Build the list of rounds to scrape (regular + finals)
    rounds_to_scrape: list[int | str] = list(REGULAR_ROUNDS) + FINALS_ROUNDS

    # -----------------------------------------------------------------------
    # Step 1: Scrape round summaries and ladders for each season
    # -----------------------------------------------------------------------
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n{'=' * 60}")
        print(f"  Season {year}")
        print(f"{'=' * 60}")
        t_season = time.time()

        # --- Round summaries (matches) ---
        print(f"  Scraping round summaries for {year}...")
        round_htmls = scraper.scrape_season_round_summaries(
            year, rounds=rounds_to_scrape
        )

        season_match_count = 0
        for round_id, html in round_htmls.items():
            try:
                parsed = parse_round_summary(html, year, round_id)
                all_matches.extend(parsed)
                season_match_count += len(parsed)
            except Exception as exc:
                logger.error(
                    "Error parsing round summary for %d round %s: %s",
                    year, round_id, exc,
                )

        print(f"  -> {season_match_count} matches from {len(round_htmls)} rounds")

        # --- Ladders (regular season only) ---
        print(f"  Scraping ladders for {year}...")
        ladder_htmls = scraper.scrape_season_round_ladders(year)

        season_ladder_count = 0
        for round_num, html in ladder_htmls.items():
            try:
                parsed_ladder = parse_round_ladder(html, year, round_num)
                all_ladders.extend(parsed_ladder)
                season_ladder_count += len(parsed_ladder)
            except Exception as exc:
                logger.error(
                    "Error parsing ladder for %d round %s: %s",
                    year, round_num, exc,
                )

        print(f"  -> {season_ladder_count} ladder rows from {len(ladder_htmls)} rounds")

        elapsed = time.time() - t_season
        print(f"  Season {year} completed in {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Step 2: Build DataFrames
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Building DataFrames")
    print(f"{'=' * 60}")

    matches_df = pd.DataFrame(all_matches) if all_matches else pd.DataFrame()
    ladders_df = pd.DataFrame(all_ladders) if all_ladders else pd.DataFrame()

    print(f"  Matches: {len(matches_df)} rows")
    print(f"  Ladders: {len(ladders_df)} rows")

    # -----------------------------------------------------------------------
    # Step 3: Standardise team names
    # -----------------------------------------------------------------------
    print("\n  Standardising team names...")

    if not matches_df.empty:
        standardise_team_columns(matches_df, ["home_team", "away_team"])

    if not ladders_df.empty:
        standardise_team_columns(ladders_df, ["team"])

    # -----------------------------------------------------------------------
    # Step 3b: Fix home/away using venue data
    # -----------------------------------------------------------------------
    print("\n  Fixing home/away team assignment (RLP lists winner first)...")
    if not matches_df.empty:
        from processing.venue_home_fix import fix_home_away
        matches_df = fix_home_away(matches_df)

    # -----------------------------------------------------------------------
    # Step 4: Load odds data
    # -----------------------------------------------------------------------
    print("\n  Loading odds data...")
    try:
        odds_df = load_odds(standardise_teams=True, parse_dates=True)
        print(f"  Odds: {len(odds_df)} rows")
    except FileNotFoundError as exc:
        logger.warning("Odds file not found: %s", exc)
        odds_df = pd.DataFrame()
    except Exception as exc:
        logger.error("Failed to load odds: %s", exc)
        odds_df = pd.DataFrame()

    # -----------------------------------------------------------------------
    # Step 5: Save to Parquet
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Saving Parquet files")
    print(f"{'=' * 60}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Fix mixed-type columns before Parquet serialization.
    # The 'round' column has both int (1-27) and str ('qualif-final', etc.),
    # which PyArrow cannot handle. Convert to string uniformly.
    for df in [matches_df, ladders_df]:
        if not df.empty and "round" in df.columns:
            df["round"] = df["round"].astype(str)

    # For matches: drop columns with complex types (lists/dicts) that
    # don't serialise cleanly to Parquet. We keep the core columns.
    if not matches_df.empty:
        # Identify columns that contain list/dict data
        drop_cols = []
        for col in matches_df.columns:
            sample = matches_df[col].dropna().head(5)
            if any(isinstance(v, (list, dict)) for v in sample):
                drop_cols.append(col)

        if drop_cols:
            print(f"  Note: Dropping {len(drop_cols)} complex columns from matches: {drop_cols}")
            matches_clean = matches_df.drop(columns=drop_cols)
        else:
            matches_clean = matches_df

        matches_path = PROCESSED_DIR / "matches.parquet"
        matches_clean.to_parquet(matches_path, index=False, engine="pyarrow")
        print(f"  Saved matches -> {matches_path}  ({len(matches_clean)} rows, {len(matches_clean.columns)} cols)")
    else:
        print("  Skipping matches (empty)")

    if not ladders_df.empty:
        ladders_path = PROCESSED_DIR / "ladders.parquet"
        ladders_df.to_parquet(ladders_path, index=False, engine="pyarrow")
        print(f"  Saved ladders -> {ladders_path}  ({len(ladders_df)} rows, {len(ladders_df.columns)} cols)")
    else:
        print("  Skipping ladders (empty)")

    if not odds_df.empty:
        # Fix columns with types that PyArrow can't handle (e.g. datetime.time).
        import datetime as _dt
        for col in odds_df.columns:
            if odds_df[col].dtype == object:
                # Check actual values for non-serializable types
                non_null = odds_df[col].dropna()
                if len(non_null) > 0:
                    first_val = non_null.iloc[0]
                    if isinstance(first_val, (_dt.time, _dt.timedelta)):
                        odds_df[col] = odds_df[col].astype(str)

        odds_path = PROCESSED_DIR / "odds.parquet"
        odds_df.to_parquet(odds_path, index=False, engine="pyarrow")
        print(f"  Saved odds -> {odds_path}  ({len(odds_df)} rows, {len(odds_df.columns)} cols)")
    else:
        print("  Skipping odds (empty)")

    # -----------------------------------------------------------------------
    # Step 6: Verification
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Verification")
    print(f"{'=' * 60}")

    if not matches_df.empty:
        print("\n  --- Matches ---")
        print(f"  Shape: {matches_clean.shape}")
        print(f"  Columns: {list(matches_clean.columns)}")

        if "year" in matches_clean.columns:
            print("\n  Matches per season:")
            season_counts = matches_clean.groupby("year").size()
            for yr, count in season_counts.items():
                print(f"    {yr}: {count} matches")

        if "home_team" in matches_clean.columns:
            all_teams = set(matches_clean["home_team"].dropna().unique())
            if "away_team" in matches_clean.columns:
                all_teams |= set(matches_clean["away_team"].dropna().unique())
            print(f"\n  Unique teams in matches: {len(all_teams)}")
            for t in sorted(all_teams):
                print(f"    - {t}")

        print(f"\n  Sample data (first 5 rows, key columns):")
        key_cols = [c for c in ["year", "round", "home_team", "away_team",
                                "home_score", "away_score", "venue", "attendance"]
                    if c in matches_clean.columns]
        print(matches_clean[key_cols].head().to_string(index=False))

    if not ladders_df.empty:
        print("\n  --- Ladders ---")
        print(f"  Shape: {ladders_df.shape}")
        print(f"  Columns: {list(ladders_df.columns)}")

        if "team" in ladders_df.columns:
            ladder_teams = set(ladders_df["team"].dropna().unique())
            print(f"\n  Unique teams in ladders: {len(ladder_teams)}")
            for t in sorted(ladder_teams):
                print(f"    - {t}")

        if "year" in ladders_df.columns:
            print("\n  Ladder rows per season:")
            ladder_season_counts = ladders_df.groupby("year").size()
            for yr, count in ladder_season_counts.items():
                print(f"    {yr}: {count} rows")

        print(f"\n  Sample ladder data (first 5 rows):")
        key_cols = [c for c in ["year", "round", "position", "team",
                                "played", "won", "lost", "competition_points"]
                    if c in ladders_df.columns]
        print(ladders_df[key_cols].head().to_string(index=False))

    if not odds_df.empty:
        print("\n  --- Odds ---")
        print(f"  Shape: {odds_df.shape}")
        print(f"  Columns: {list(odds_df.columns)}")

        if "home_team" in odds_df.columns:
            odds_teams = set(odds_df["home_team"].dropna().unique())
            if "away_team" in odds_df.columns:
                odds_teams |= set(odds_df["away_team"].dropna().unique())
            print(f"\n  Unique teams in odds: {len(odds_teams)}")
            for t in sorted(odds_teams):
                print(f"    - {t}")

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline completed in {total_elapsed:.1f}s")
    print(f"  All files saved to: {PROCESSED_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
