"""
Feature-building pipeline: raw/processed data -> feature matrix.

Loads processed Parquet files, cleans them, links datasets together,
computes Elo ratings, rolling stats, and builds a versioned feature
matrix ready for model training.

Usage
-----
::

    # Build v2 features (default: Elo + rolling form + H2H + venue)
    python -m pipelines.build_features

    # Build v4 features (includes lineup + odds)
    python -m pipelines.build_features --feature-version v4

    # Custom output path
    python -m pipelines.build_features --feature-version v3 \\
        --output-path data/features/features_v3_custom.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config.settings import (
    FEATURES_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    START_YEAR,
    END_YEAR,
)
from processing.data_cleaning import (
    clean_matches,
    clean_ladder,
    clean_lineups,
    clean_odds,
)
from processing.data_linking import build_master_dataset
from processing.feature_engineering import build_all_features
from processing.target_encoding import (
    create_target,
    target_encode_categoricals,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data loading helpers
# ============================================================================

def _load_parquet(name: str, directory: Path) -> pd.DataFrame:
    """Load a Parquet file from the processed directory.

    Returns an empty DataFrame if the file does not exist.
    """
    path = directory / f"{name}.parquet"
    if not path.exists():
        logger.warning("File not found: %s -- returning empty DataFrame.", path)
        return pd.DataFrame()
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded %s: %d rows, %d cols.", path.name, len(df), len(df.columns))
    return df


# ============================================================================
# Core pipeline
# ============================================================================

def build_feature_matrix(
    feature_version: str = "v2",
    processed_dir: Path | None = None,
    output_path: Path | None = None,
    draw_handling: str = "exclude",
    target_encode: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Execute the full feature-building pipeline.

    Parameters
    ----------
    feature_version : str
        Which feature set to build.  One of ``"v1"`` through ``"v4"``.
        See ``processing.feature_engineering.build_all_features`` for
        descriptions of what each version includes.
    processed_dir : Path, optional
        Directory containing the processed Parquet files.  Defaults to
        ``data/processed/``.
    output_path : Path, optional
        Where to save the final feature Parquet.  Defaults to
        ``data/features/features_{version}.parquet``.
    draw_handling : str
        How to handle draws when creating the target variable.
        ``"exclude"`` (default) sets draws to NaN (they are dropped
        before training).
    target_encode : bool
        Whether to target-encode categorical columns (venue, team names).
        Default True.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (features_df, target) -- the feature matrix and the binary target.
    """
    proc_dir = processed_dir or PROCESSED_DIR

    # ------------------------------------------------------------------
    # 1. Load processed Parquet files
    # ------------------------------------------------------------------
    print("\n  [1/6] Loading processed data...")
    matches_raw = _load_parquet("matches", proc_dir)
    lineups_raw = _load_parquet("lineups", proc_dir)
    ladders_raw = _load_parquet("ladders", proc_dir)
    odds_raw = _load_parquet("odds", proc_dir)
    players_raw = _load_parquet("players", proc_dir)

    if matches_raw.empty:
        raise FileNotFoundError(
            f"No matches data found in {proc_dir}. "
            "Run `python -m pipelines.scrape_all` first."
        )

    print(f"    Matches  : {len(matches_raw):>6,} rows")
    print(f"    Lineups  : {len(lineups_raw):>6,} rows")
    print(f"    Ladders  : {len(ladders_raw):>6,} rows")
    print(f"    Odds     : {len(odds_raw):>6,} rows")
    print(f"    Players  : {len(players_raw):>6,} rows")

    # ------------------------------------------------------------------
    # 2. Clean each dataset
    # ------------------------------------------------------------------
    print("\n  [2/6] Cleaning data...")
    t0 = time.time()

    matches = clean_matches(matches_raw)
    print(f"    Matches cleaned: {len(matches):>6,} rows ({time.time()-t0:.1f}s)")

    ladders = clean_ladder(ladders_raw) if not ladders_raw.empty else pd.DataFrame()
    lineups = clean_lineups(lineups_raw) if not lineups_raw.empty else pd.DataFrame()
    odds = clean_odds(odds_raw) if not odds_raw.empty else pd.DataFrame()
    players = players_raw  # players are used as-is for career lookups

    # ------------------------------------------------------------------
    # 3. Create target variable
    # ------------------------------------------------------------------
    print("\n  [3/6] Creating target variable...")
    target = create_target(matches, draw_handling=draw_handling)

    n_home = int((target == 1.0).sum())
    n_away = int((target == 0.0).sum())
    n_na = int(target.isna().sum())
    print(f"    Home wins : {n_home:>5,}")
    print(f"    Away wins : {n_away:>5,}")
    print(f"    Draws/NaN : {n_na:>5,}")

    # ------------------------------------------------------------------
    # 4. Build features
    # ------------------------------------------------------------------
    print(f"\n  [4/6] Building feature set '{feature_version}'...")
    t0 = time.time()

    features_df = build_all_features(
        matches=matches,
        lineups=lineups if not lineups.empty else None,
        ladders=ladders if not ladders.empty else None,
        players=players if not players.empty else None,
        odds=odds if not odds.empty else None,
        feature_version=feature_version,
    )

    elapsed = time.time() - t0
    print(f"    Feature matrix: {features_df.shape[0]:,} rows x {features_df.shape[1]:,} cols ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Target-encode categoricals (optional)
    # ------------------------------------------------------------------
    if target_encode:
        print("\n  [5/6] Target-encoding categorical columns...")
        cat_cols = [c for c in ["venue", "home_team", "away_team"] if c in features_df.columns]
        if cat_cols:
            # Only encode rows with valid targets
            valid_mask = target.notna()
            features_df, _, encoder = target_encode_categoricals(
                train_df=features_df,
                target=target,
                columns=cat_cols,
            )
            print(f"    Encoded columns: {cat_cols}")
        else:
            print("    No categorical columns found to encode.")
    else:
        print("\n  [5/6] Skipping target encoding (disabled).")

    # ------------------------------------------------------------------
    # 6. Add a year column for backtesting and drop target leakage cols
    # ------------------------------------------------------------------
    print("\n  [6/6] Finalising feature matrix...")

    # Ensure we have a 'year' column for the backtester
    # (clean_matches produces 'season', but WalkForwardBacktester defaults to 'year')
    if "year" not in features_df.columns and "season" in features_df.columns:
        features_df["year"] = pd.to_numeric(features_df["season"], errors="coerce").astype("Int64")
    elif "year" not in features_df.columns and "season_year" in features_df.columns:
        features_df["year"] = features_df["season_year"]

    # Ensure we have a 'round' column
    if "round" not in features_df.columns and "round_number" in features_df.columns:
        features_df["round"] = features_df["round_number"]

    # Add target as a column in the saved file for convenience
    features_df["target_home_win"] = target.values

    # Identify columns that would constitute target leakage
    leakage_cols = [
        "home_score", "away_score", "margin", "home_win",
        "is_draw", "result", "winner",
    ]
    cols_to_drop = [c for c in leakage_cols if c in features_df.columns]
    if cols_to_drop:
        features_df = features_df.drop(columns=cols_to_drop)
        print(f"    Dropped leakage columns: {cols_to_drop}")

    # Report final shape
    n_numeric = features_df.select_dtypes(include=[np.number]).shape[1]
    print(f"    Final shape: {features_df.shape[0]:,} rows x {features_df.shape[1]:,} cols")
    print(f"    Numeric features: {n_numeric}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if output_path is None:
        output_path = FEATURES_DIR / f"features_{feature_version}.parquet"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"\n    Saved to: {output_path}")

    return features_df, target


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="pipelines.build_features",
        description="Build a versioned feature matrix from processed NRL data.",
    )
    parser.add_argument(
        "--feature-version",
        type=str,
        default="v2",
        choices=["v1", "v2", "v3", "v4"],
        help="Feature set version to build (default: v2).",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help=f"Directory containing processed Parquet files (default: {PROCESSED_DIR}).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for the feature Parquet (default: data/features/features_{version}.parquet).",
    )
    parser.add_argument(
        "--draw-handling",
        type=str,
        default="exclude",
        choices=["exclude", "home", "away", "half"],
        help="How to handle drawn matches in the target variable (default: exclude).",
    )
    parser.add_argument(
        "--no-target-encode",
        action="store_true",
        default=False,
        help="Skip target encoding of categorical columns.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the feature-building pipeline."""
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\nNRL Feature Builder")
    print(f"  Feature version : {args.feature_version}")
    print(f"  Draw handling   : {args.draw_handling}")
    print(f"  Target encoding : {'No' if args.no_target_encode else 'Yes'}")

    t_start = time.time()

    features_df, target = build_feature_matrix(
        feature_version=args.feature_version,
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        output_path=Path(args.output_path) if args.output_path else None,
        draw_handling=args.draw_handling,
        target_encode=not args.no_target_encode,
    )

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed time: {elapsed:.1f}s")
    print("  Done.\n")


if __name__ == "__main__":
    main()
