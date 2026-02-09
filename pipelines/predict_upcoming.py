"""
Generate predictions for upcoming NRL matches.

Loads a trained model from the registry, computes features for upcoming
matches, generates win probabilities, and outputs predictions as a
formatted table, CSV file, and optional HTML report.

Usage
-----
::

    # Predict next round with the best model
    python -m pipelines.predict_upcoming --year 2026 --round 1

    # Use a specific model from the registry
    python -m pipelines.predict_upcoming --year 2026 --round 5 \\
        --model-name xgboost_v2 --model-version 3

    # Output predictions to a specific CSV file
    python -m pipelines.predict_upcoming --year 2026 --round 1 \\
        --output-csv outputs/predictions_r1.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config.settings import (
    FEATURES_DIR,
    OUTPUTS_DIR,
    PREDICT_YEAR,
    PROCESSED_DIR,
)
from modelling.model_registry import (
    get_model_metadata,
    list_models as list_registry_models,
    load_model,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Prediction helpers
# ============================================================================

def _find_best_model() -> tuple[str, str | None]:
    """Scan the model registry and return (name, version) of the best model.

    The "best" model is determined by the lowest ``log_loss`` in the stored
    metadata, or falls back to the most recently saved model.

    Returns
    -------
    tuple[str, str | None]
        (model_name, version)

    Raises
    ------
    FileNotFoundError
        If no models are found in the registry.
    """
    all_models = list_registry_models()
    if not all_models:
        raise FileNotFoundError(
            "No models found in the registry. "
            "Run `python -m pipelines.train_and_evaluate` first."
        )

    # Try to find the model flagged as best
    best_entry = None
    best_log_loss = float("inf")

    for entry in all_models:
        user_meta = entry.get("user_metadata", {})
        if user_meta.get("is_best", False):
            ll = user_meta.get("log_loss", float("inf"))
            if ll < best_log_loss:
                best_log_loss = ll
                best_entry = entry

    # Fallback: most recently saved model
    if best_entry is None:
        # Sort by saved_at descending
        sorted_models = sorted(
            all_models,
            key=lambda x: x.get("saved_at", ""),
            reverse=True,
        )
        best_entry = sorted_models[0]

    return best_entry["name"], best_entry.get("version")


def _load_upcoming_matches(
    year: int,
    round_num: int | str,
    processed_dir: Path,
) -> pd.DataFrame:
    """Load or construct a DataFrame of upcoming matches.

    First checks for a manually prepared ``upcoming.parquet`` or
    ``upcoming.csv`` file in ``data/processed/``.  Falls back to loading
    the main matches file and filtering for unplayed matches (where
    scores are NaN).

    Parameters
    ----------
    year : int
        Season year.
    round_num : int or str
        Round identifier.
    processed_dir : Path
        Directory containing processed data files.

    Returns
    -------
    pd.DataFrame
        Upcoming matches with at least ``home_team`` and ``away_team``.
    """
    # Check for a dedicated upcoming matches file
    for suffix in ("parquet", "csv"):
        upcoming_path = processed_dir / f"upcoming.{suffix}"
        if upcoming_path.exists():
            if suffix == "parquet":
                df = pd.read_parquet(upcoming_path, engine="pyarrow")
            else:
                df = pd.read_csv(upcoming_path)
            logger.info("Loaded upcoming matches from %s: %d rows.", upcoming_path, len(df))
            # Filter to the requested year and round
            if "year" in df.columns:
                df = df[df["year"] == year]
            elif "season" in df.columns:
                df = df[df["season"] == year]
            if "round" in df.columns:
                df = df[df["round"].astype(str) == str(round_num)]
            if not df.empty:
                return df

    # Fallback: look at matches with missing scores
    matches_path = processed_dir / "matches.parquet"
    if matches_path.exists():
        matches = pd.read_parquet(matches_path, engine="pyarrow")

        # Filter to the year and round
        year_col = "year" if "year" in matches.columns else "season"
        if year_col in matches.columns:
            matches = matches[matches[year_col] == year]
        if "round" in matches.columns:
            matches = matches[matches["round"].astype(str) == str(round_num)]

        # Keep rows without scores (upcoming matches)
        if "home_score" in matches.columns:
            upcoming = matches[matches["home_score"].isna()]
            if not upcoming.empty:
                return upcoming

        # If all matches have scores, return them anyway (user may want
        # to generate predictions for historical comparison)
        if not matches.empty:
            logger.info(
                "All matches for %d Round %s have scores. "
                "Returning them for prediction (historical comparison mode).",
                year, round_num,
            )
            return matches

    raise FileNotFoundError(
        f"No upcoming matches found for {year} Round {round_num}. "
        f"Create an 'upcoming.csv' or 'upcoming.parquet' file in {processed_dir} "
        "with columns: home_team, away_team, venue, date "
        "(and optionally home_odds, away_odds)."
    )


def _build_upcoming_features(
    upcoming_df: pd.DataFrame,
    feature_cols: list[str],
    feature_version: str,
    features_dir: Path,
    processed_dir: Path,
) -> pd.DataFrame:
    """Build the feature matrix for upcoming matches.

    Uses the full historical feature pipeline to compute features that
    require historical context (Elo, rolling form, etc.).  The upcoming
    matches are appended to the historical data before feature
    computation, and only the upcoming rows are returned.

    Parameters
    ----------
    upcoming_df : pd.DataFrame
        Upcoming match records.
    feature_cols : list[str]
        The list of feature columns the model expects.
    feature_version : str
        Feature set version used for training.
    features_dir : Path
        Directory containing the pre-built feature file.
    processed_dir : Path
        Directory containing processed data files.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per upcoming match, columns aligned
        to *feature_cols*.
    """
    # Strategy 1: Try to use build_all_features on historical + upcoming
    try:
        from processing.data_cleaning import clean_matches, clean_ladder, clean_odds
        from processing.feature_engineering import build_all_features

        # Load historical matches
        hist_path = processed_dir / "matches.parquet"
        if hist_path.exists():
            hist_matches = pd.read_parquet(hist_path, engine="pyarrow")
            hist_matches = clean_matches(hist_matches)

            # Append upcoming matches (with NaN scores)
            upcoming_clean = upcoming_df.copy()
            for col in ["home_score", "away_score"]:
                if col not in upcoming_clean.columns:
                    upcoming_clean[col] = np.nan

            combined = pd.concat([hist_matches, upcoming_clean], ignore_index=True)

            # Load supporting data
            ladders = _load_parquet_safe("ladders", processed_dir)
            odds = _load_parquet_safe("odds", processed_dir)
            lineups = _load_parquet_safe("lineups", processed_dir)
            players = _load_parquet_safe("players", processed_dir)

            # Clean
            if not ladders.empty:
                ladders = clean_ladder(ladders)
            if not odds.empty:
                odds = clean_odds(odds)

            all_features = build_all_features(
                matches=combined,
                lineups=lineups if not lineups.empty else None,
                ladders=ladders if not ladders.empty else None,
                players=players if not players.empty else None,
                odds=odds if not odds.empty else None,
                feature_version=feature_version,
            )

            # Extract only the upcoming rows (last N rows = len(upcoming_df))
            upcoming_features = all_features.tail(len(upcoming_df)).copy()

            # Align columns to what the model expects
            for col in feature_cols:
                if col not in upcoming_features.columns:
                    upcoming_features[col] = np.nan

            return upcoming_features[feature_cols]

    except Exception as exc:
        logger.warning(
            "Full feature pipeline failed for upcoming matches: %s. "
            "Falling back to direct column mapping.",
            exc,
        )

    # Strategy 2: Simple fallback -- fill missing feature columns with NaN
    logger.info("Using simple fallback for feature construction.")
    result = pd.DataFrame(index=range(len(upcoming_df)))
    for col in feature_cols:
        if col in upcoming_df.columns:
            result[col] = upcoming_df[col].values
        else:
            result[col] = np.nan

    return result


def _load_parquet_safe(name: str, directory: Path) -> pd.DataFrame:
    """Load a Parquet file; return empty DataFrame if not found."""
    path = directory / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path, engine="pyarrow")
    return pd.DataFrame()


def _format_predictions_table(predictions_df: pd.DataFrame) -> str:
    """Format predictions as a readable console table."""
    lines = []
    header = (
        f"{'Home':<25s} {'Away':<25s} {'Home %':>8s} {'Away %':>8s} "
        f"{'Winner':<25s} {'Confidence':>10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in predictions_df.iterrows():
        home = str(row.get("home_team", "???"))[:24]
        away = str(row.get("away_team", "???"))[:24]
        home_prob = row.get("y_prob", 0.5)
        away_prob = 1.0 - home_prob
        winner = home if home_prob >= 0.5 else away
        confidence = max(home_prob, away_prob)

        line = (
            f"{home:<25s} {away:<25s} {home_prob:>7.1%} {away_prob:>8.1%} "
            f"{winner:<25s} {confidence:>9.1%}"
        )

        # Add odds-related info if available
        if "home_odds" in row.index and pd.notna(row.get("home_odds")):
            implied = 1.0 / row["home_odds"] if row["home_odds"] > 0 else 0.0
            edge = home_prob - implied
            line += f"  Edge: {edge:+.1%}"

        lines.append(line)

    return "\n".join(lines)


# ============================================================================
# Core pipeline
# ============================================================================

def predict_upcoming(
    year: int,
    round_num: int | str,
    model_name: str | None = None,
    model_version: str | None = None,
    output_csv: Path | None = None,
    generate_html: bool = True,
    processed_dir: Path | None = None,
    features_dir: Path | None = None,
) -> pd.DataFrame:
    """Generate predictions for upcoming matches.

    Parameters
    ----------
    year : int
        Season year.
    round_num : int or str
        Round identifier.
    model_name : str, optional
        Model name in the registry.  If None, auto-selects the best model.
    model_version : str, optional
        Model version.  If None, loads the latest version.
    output_csv : Path, optional
        Where to save the predictions CSV.
    generate_html : bool
        Whether to generate an HTML prediction report.
    processed_dir : Path, optional
        Directory with processed data.
    features_dir : Path, optional
        Directory with feature files.

    Returns
    -------
    pd.DataFrame
        Prediction DataFrame with columns: home_team, away_team, y_prob,
        y_pred, confidence, and optionally odds/edge columns.
    """
    proc_dir = processed_dir or PROCESSED_DIR
    feat_dir = features_dir or FEATURES_DIR

    # ------------------------------------------------------------------
    # 1. Load model from registry
    # ------------------------------------------------------------------
    print(f"\n  [1/4] Loading model...")

    if model_name is None:
        model_name, model_version = _find_best_model()
        print(f"    Auto-selected: {model_name} v{model_version}")
    else:
        print(f"    Requested: {model_name}" +
              (f" v{model_version}" if model_version else " (latest)"))

    model = load_model(model_name, version=model_version)
    metadata = get_model_metadata(model_name, version=model_version)
    user_meta = metadata.get("user_metadata", {})

    feature_version = user_meta.get("feature_version", "v2")
    feature_cols = user_meta.get("feature_columns", [])

    print(f"    Model type      : {metadata.get('model_type', 'unknown')}")
    print(f"    Feature version : {feature_version}")
    print(f"    N features      : {len(feature_cols)}")
    if "accuracy" in user_meta:
        print(f"    Backtest acc    : {user_meta['accuracy']:.4f}")
    if "log_loss" in user_meta:
        print(f"    Backtest LL     : {user_meta['log_loss']:.4f}")

    # ------------------------------------------------------------------
    # 2. Load upcoming matches
    # ------------------------------------------------------------------
    print(f"\n  [2/4] Loading upcoming matches for {year} Round {round_num}...")

    upcoming_df = _load_upcoming_matches(year, round_num, proc_dir)
    print(f"    Found {len(upcoming_df)} matches.")

    if upcoming_df.empty:
        print("    No matches found. Exiting.")
        return pd.DataFrame()

    for _, row in upcoming_df.iterrows():
        home = row.get("home_team", "???")
        away = row.get("away_team", "???")
        venue = row.get("venue", "")
        print(f"      {home} vs {away}" + (f" @ {venue}" if venue else ""))

    # ------------------------------------------------------------------
    # 3. Build features for upcoming matches
    # ------------------------------------------------------------------
    print(f"\n  [3/4] Computing features...")

    if not feature_cols:
        # If feature columns were not stored in metadata, try to infer from
        # the feature file
        feat_path = feat_dir / f"features_{feature_version}.parquet"
        if feat_path.exists():
            sample = pd.read_parquet(feat_path, engine="pyarrow", columns=None)
            # Infer numeric feature columns
            meta_cols = {"year", "round", "season", "season_year", "date", "kickoff",
                         "home_team", "away_team", "venue", "match_id", "target_home_win"}
            feature_cols = [
                c for c in sample.select_dtypes(include=[np.number]).columns
                if c not in meta_cols
            ]
            logger.info("Inferred %d feature columns from %s.", len(feature_cols), feat_path)

    if not feature_cols:
        raise ValueError(
            "Cannot determine feature columns. Ensure the model metadata "
            "contains 'feature_columns' or that the feature Parquet exists."
        )

    X_upcoming = _build_upcoming_features(
        upcoming_df=upcoming_df,
        feature_cols=feature_cols,
        feature_version=feature_version,
        features_dir=feat_dir,
        processed_dir=proc_dir,
    )

    n_missing = X_upcoming.isna().all(axis=0).sum()
    if n_missing > 0:
        logger.warning(
            "%d feature column(s) are entirely NaN for upcoming matches.", n_missing
        )

    # Fill remaining NaNs with 0 (reasonable default for missing features)
    X_upcoming = X_upcoming.fillna(0.0)

    print(f"    Feature matrix: {X_upcoming.shape[0]} rows x {X_upcoming.shape[1]} cols")

    # ------------------------------------------------------------------
    # 4. Generate predictions
    # ------------------------------------------------------------------
    print(f"\n  [4/4] Generating predictions...")

    y_prob_raw = model.predict_proba(X_upcoming)
    if hasattr(y_prob_raw, "ndim") and y_prob_raw.ndim == 2:
        y_prob = y_prob_raw[:, 1]
    else:
        y_prob = np.asarray(y_prob_raw)

    y_pred = (y_prob >= 0.5).astype(int)

    # Build output DataFrame
    predictions = pd.DataFrame({
        "home_team": upcoming_df["home_team"].values,
        "away_team": upcoming_df["away_team"].values,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "predicted_winner": np.where(
            y_prob >= 0.5,
            upcoming_df["home_team"].values,
            upcoming_df["away_team"].values,
        ),
        "confidence": np.maximum(y_prob, 1.0 - y_prob),
    })

    # Add venue if available
    if "venue" in upcoming_df.columns:
        predictions["venue"] = upcoming_df["venue"].values

    # Add odds info if available
    for col in ["home_odds", "away_odds", "h2h_home", "h2h_away"]:
        if col in upcoming_df.columns:
            predictions[col] = upcoming_df[col].values

    # Calculate edge against odds if available
    if "home_odds" in predictions.columns:
        predictions["implied_prob_home"] = 1.0 / predictions["home_odds"].replace(0, np.nan)
        predictions["model_edge"] = predictions["y_prob"] - predictions["implied_prob_home"]
        predictions["value_bet"] = predictions["model_edge"] > 0.05

    # ------------------------------------------------------------------
    # Display predictions
    # ------------------------------------------------------------------
    print(f"\n  {'='*80}")
    print(f"  Predictions: {year} Round {round_num}")
    print(f"  Model: {model_name} v{model_version or 'latest'}")
    print(f"  {'='*80}")
    print()
    print(_format_predictions_table(predictions))
    print()

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    if output_csv is None:
        output_csv = OUTPUTS_DIR / "predictions" / f"predictions_{year}_round_{round_num}.csv"
    else:
        output_csv = Path(output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    print(f"  Predictions saved to: {output_csv}")

    # Generate HTML report
    if generate_html:
        try:
            from evaluation.reports import generate_prediction_report

            report_html = generate_prediction_report(
                predictions_df=predictions,
                round_num=round_num,
                year=year,
                save=True,
            )
            report_path = OUTPUTS_DIR / "reports" / f"predictions_{year}_round_{round_num}.html"
            print(f"  HTML report saved to: {report_path}")
        except Exception as exc:
            logger.error("HTML report generation failed: %s", exc)
            print(f"  -> HTML report generation FAILED: {exc}")

    return predictions


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="pipelines.predict_upcoming",
        description="Generate NRL match predictions for upcoming rounds.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=PREDICT_YEAR,
        help=f"Season year (default: {PREDICT_YEAR}).",
    )
    parser.add_argument(
        "--round",
        type=str,
        default="1",
        help="Round number or finals slug (default: 1).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name from the registry. Auto-selects best if not specified.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Model version. Loads latest if not specified.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path for predictions.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        default=False,
        help="Skip HTML report generation.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help=f"Directory with processed data (default: {PROCESSED_DIR}).",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help=f"Directory with feature files (default: {FEATURES_DIR}).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the prediction pipeline."""
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse round number (could be int or string like "grand-final")
    try:
        round_num: int | str = int(args.round)
    except ValueError:
        round_num = args.round

    print(f"\nNRL Match Prediction")
    print(f"  Year       : {args.year}")
    print(f"  Round      : {round_num}")
    print(f"  Model      : {args.model_name or '(auto-select best)'}")
    if args.model_version:
        print(f"  Version    : {args.model_version}")

    t_start = time.time()

    predictions = predict_upcoming(
        year=args.year,
        round_num=round_num,
        model_name=args.model_name,
        model_version=args.model_version,
        output_csv=Path(args.output_csv) if args.output_csv else None,
        generate_html=not args.no_html,
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        features_dir=Path(args.features_dir) if args.features_dir else None,
    )

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed time: {elapsed:.1f}s")
    print("  Done.\n")


if __name__ == "__main__":
    main()
