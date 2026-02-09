"""
End-to-end scrape orchestrator for NRL match data.

Downloads HTML pages from Rugby League Project, parses them into structured
records (matches, lineups, ladders, players), loads odds from the
AusSportsBetting Excel file, standardises all team names, and persists
everything as Parquet files under ``data/processed/``.

Usage
-----
::

    # Scrape all seasons with defaults (2013-2025)
    python -m pipelines.scrape_all

    # Custom year range, include finals and player data
    python -m pipelines.scrape_all --start-year 2020 --end-year 2025 \\
        --include-finals --include-players

    # Skip player scraping for a faster run
    python -m pipelines.scrape_all --start-year 2023 --end-year 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config.settings import (
    END_YEAR,
    FINALS_ROUNDS,
    PROCESSED_DIR,
    RAW_DIR,
    REGULAR_ROUNDS,
    START_YEAR,
)
from config.team_mappings import standardise_team_name
from scraping.rlp_scraper import RLPScraper
from scraping.rlp_match_parser import parse_round_summary
from scraping.rlp_ladder_parser import parse_round_ladder
from scraping.rlp_player_parser import parse_season_players

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _standardise_team_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply ``standardise_team_name`` to the specified columns, in-place.

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


def _save_parquet(df: pd.DataFrame, name: str, directory: Path) -> Path:
    """Save a DataFrame as a Parquet file and log the result."""
    path = directory / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %s -> %s  (%d rows, %d cols)", name, path, len(df), len(df.columns))
    return path


# ============================================================================
# Core pipeline
# ============================================================================

def scrape_and_parse(
    start_year: int,
    end_year: int,
    include_finals: bool = True,
    include_players: bool = False,
    include_stats: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the full scrape-and-parse pipeline.

    Parameters
    ----------
    start_year : int
        First season to scrape (inclusive).
    end_year : int
        Last season to scrape (inclusive).
    include_finals : bool
        If True, scrape finals rounds in addition to regular season rounds.
    include_players : bool
        If True, scrape season-level player listings.
    include_stats : bool
        If True, also load NRL advanced match stats (from S3 JSON source).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``"matches"``, ``"lineups"``, ``"ladders"``,
        ``"odds"``, and optionally ``"players"``.
    """
    scraper = RLPScraper(use_cache=True, show_progress=True)

    all_matches: list[dict] = []
    all_ladders: list[dict] = []
    all_players: list[dict] = []

    for year in range(start_year, end_year + 1):
        print(f"\n{'='*60}")
        print(f"  Scraping season {year}")
        print(f"{'='*60}")
        t0 = time.time()

        # --- Round summaries (matches + lineups) --------------------------
        rounds_to_scrape: list[int | str] = list(REGULAR_ROUNDS)
        if include_finals:
            rounds_to_scrape.extend(FINALS_ROUNDS)

        round_htmls = scraper.scrape_season_round_summaries(year, rounds=rounds_to_scrape)

        season_match_count = 0
        for round_id, html in round_htmls.items():
            parsed = parse_round_summary(html, year, round_id)
            all_matches.extend(parsed)
            season_match_count += len(parsed)

        print(f"  -> {season_match_count} matches parsed from {len(round_htmls)} rounds.")

        # --- Ladders (regular season only) --------------------------------
        ladder_htmls = scraper.scrape_season_round_ladders(year)
        season_ladder_count = 0
        for round_num, html in ladder_htmls.items():
            parsed_ladder = parse_round_ladder(html, year, round_num)
            all_ladders.extend(parsed_ladder)
            season_ladder_count += len(parsed_ladder)

        print(f"  -> {season_ladder_count} ladder rows from {len(ladder_htmls)} rounds.")

        # --- Players (optional) -------------------------------------------
        if include_players:
            player_html = scraper.scrape_season_players(year)
            if player_html is not None:
                parsed_players = parse_season_players(player_html, year)
                all_players.extend(parsed_players)
                print(f"  -> {len(parsed_players)} player records.")
            else:
                print(f"  -> Player page not available for {year}.")

        elapsed = time.time() - t0
        print(f"  Season {year} completed in {elapsed:.1f}s.")

    # --- Build DataFrames -------------------------------------------------
    print(f"\n{'='*60}")
    print("  Building DataFrames")
    print(f"{'='*60}")

    matches_df = pd.DataFrame(all_matches) if all_matches else pd.DataFrame()
    ladders_df = pd.DataFrame(all_ladders) if all_ladders else pd.DataFrame()
    players_df = pd.DataFrame(all_players) if all_players else pd.DataFrame()

    # Separate lineups from matches if lineup columns exist
    lineups_df = _extract_lineups(matches_df) if not matches_df.empty else pd.DataFrame()

    # --- Standardise team names -------------------------------------------
    print("  Standardising team names...")
    if not matches_df.empty:
        _standardise_team_columns(matches_df, ["home_team", "away_team"])
    if not lineups_df.empty:
        _standardise_team_columns(lineups_df, ["team"])
    if not ladders_df.empty:
        _standardise_team_columns(ladders_df, ["team"])
    if not players_df.empty and "team" in players_df.columns:
        _standardise_team_columns(players_df, ["team"])

    # --- Load odds --------------------------------------------------------
    print("  Loading odds data...")
    odds_df = _load_odds_safe()

    # --- Load advanced stats (optional) -----------------------------------
    stats_df = pd.DataFrame()
    if include_stats:
        print("  Loading NRL advanced stats...")
        stats_df = _load_stats_safe(start_year, end_year)

    # --- Summary ----------------------------------------------------------
    result: dict[str, pd.DataFrame] = {
        "matches": matches_df,
        "lineups": lineups_df,
        "ladders": ladders_df,
        "odds": odds_df,
    }
    if include_players and not players_df.empty:
        result["players"] = players_df
    if include_stats and not stats_df.empty:
        result["stats"] = stats_df

    print(f"\n  Summary:")
    for name, df in result.items():
        print(f"    {name:12s}: {len(df):>6,} rows, {len(df.columns):>3} cols")

    return result


def _extract_lineups(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Extract player lineup data from match records into a separate DataFrame.

    Match records from ``parse_round_summary`` may contain columns like
    ``home_lineup``, ``away_lineup``, ``home_bench``, ``away_bench`` (lists of
    player names keyed by position number).  This function melts them into a
    long-format player-level DataFrame.
    """
    lineup_cols = [
        c for c in matches_df.columns
        if "lineup" in c.lower() or "bench" in c.lower()
    ]
    if not lineup_cols:
        return pd.DataFrame()

    rows: list[dict] = []

    for _, match in matches_df.iterrows():
        year = match.get("year", match.get("season"))
        round_id = match.get("round")

        for side in ("home", "away"):
            team = match.get(f"{side}_team")
            lineup_col = f"{side}_lineup"
            bench_col = f"{side}_bench"

            # Main lineup
            lineup_data = match.get(lineup_col)
            if isinstance(lineup_data, dict):
                for pos_str, player_name in lineup_data.items():
                    try:
                        pos = int(pos_str)
                    except (ValueError, TypeError):
                        pos = None
                    rows.append({
                        "season": year,
                        "round": round_id,
                        "team": team,
                        "home_team": match.get("home_team"),
                        "away_team": match.get("away_team"),
                        "player_name": player_name,
                        "position": pos,
                        "is_bench": False,
                    })
            elif isinstance(lineup_data, list):
                for idx, player_name in enumerate(lineup_data):
                    rows.append({
                        "season": year,
                        "round": round_id,
                        "team": team,
                        "home_team": match.get("home_team"),
                        "away_team": match.get("away_team"),
                        "player_name": player_name,
                        "position": idx + 1,
                        "is_bench": False,
                    })

            # Bench
            bench_data = match.get(bench_col)
            if isinstance(bench_data, dict):
                for pos_str, player_name in bench_data.items():
                    try:
                        pos = int(pos_str)
                    except (ValueError, TypeError):
                        pos = None
                    rows.append({
                        "season": year,
                        "round": round_id,
                        "team": team,
                        "home_team": match.get("home_team"),
                        "away_team": match.get("away_team"),
                        "player_name": player_name,
                        "position": pos,
                        "is_bench": True,
                    })
            elif isinstance(bench_data, list):
                for idx, player_name in enumerate(bench_data):
                    rows.append({
                        "season": year,
                        "round": round_id,
                        "team": team,
                        "home_team": match.get("home_team"),
                        "away_team": match.get("away_team"),
                        "player_name": player_name,
                        "position": 14 + idx,
                        "is_bench": True,
                    })

    if not rows:
        return pd.DataFrame()

    lineups_df = pd.DataFrame(rows)
    logger.info("Extracted %d lineup rows from match data.", len(lineups_df))
    return lineups_df


def _load_odds_safe() -> pd.DataFrame:
    """Attempt to load odds; return empty DataFrame on failure."""
    try:
        from scraping.odds_loader import load_odds
        return load_odds(standardise_teams=True, parse_dates=True)
    except FileNotFoundError:
        logger.warning(
            "Odds file not found at the default location. "
            "Download it from AusSportsBetting and place it in data/raw/odds/nrl.xlsx"
        )
        return pd.DataFrame()
    except Exception as exc:
        logger.error("Failed to load odds: %s", exc)
        return pd.DataFrame()


def _load_stats_safe(start_year: int, end_year: int) -> pd.DataFrame:
    """Attempt to load NRL advanced match stats; return empty DataFrame on failure."""
    try:
        from scraping.nrl_stats_loader import load_match_stats
        dfs = []
        for year in range(start_year, end_year + 1):
            try:
                df = load_match_stats(year=year, standardise_teams=True)
                dfs.append(df)
            except Exception as exc:
                logger.debug("No stats for %d: %s", year, exc)
        if dfs:
            return pd.concat(dfs, ignore_index=True)
    except Exception as exc:
        logger.error("Failed to load match stats: %s", exc)
    return pd.DataFrame()


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="pipelines.scrape_all",
        description="Scrape NRL match data from Rugby League Project and save as Parquet files.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help=f"First season to scrape (default: {START_YEAR}).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
        help=f"Last season to scrape (default: {END_YEAR}).",
    )
    parser.add_argument(
        "--include-finals",
        action="store_true",
        default=True,
        help="Include finals series rounds (default: True).",
    )
    parser.add_argument(
        "--no-finals",
        action="store_true",
        default=False,
        help="Exclude finals series rounds.",
    )
    parser.add_argument(
        "--include-players",
        action="store_true",
        default=False,
        help="Also scrape season-level player listings.",
    )
    parser.add_argument(
        "--include-stats",
        action="store_true",
        default=False,
        help="Also load NRL advanced match stats from S3 JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory for Parquet files (default: {PROCESSED_DIR}).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the scrape pipeline."""
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    include_finals = args.include_finals and not args.no_finals
    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DIR

    print(f"\nNRL Data Scraper")
    print(f"  Seasons     : {args.start_year} - {args.end_year}")
    print(f"  Finals      : {'Yes' if include_finals else 'No'}")
    print(f"  Players     : {'Yes' if args.include_players else 'No'}")
    print(f"  Stats       : {'Yes' if args.include_stats else 'No'}")
    print(f"  Output dir  : {output_dir}")

    # Run the pipeline
    t_start = time.time()
    results = scrape_and_parse(
        start_year=args.start_year,
        end_year=args.end_year,
        include_finals=include_finals,
        include_players=args.include_players,
        include_stats=args.include_stats,
    )
    elapsed_total = time.time() - t_start

    # Save outputs
    print(f"\n{'='*60}")
    print("  Saving Parquet files")
    print(f"{'='*60}")

    for name, df in results.items():
        if not df.empty:
            _save_parquet(df, name, output_dir)
        else:
            print(f"  Skipping {name} (empty).")

    print(f"\n  Total elapsed time: {elapsed_total:.1f}s")
    print(f"  All files saved to: {output_dir}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
