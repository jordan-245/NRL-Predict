"""
Training and evaluation pipeline for NRL match winner prediction.

Runs baseline models, trains classical ML models (with optional Optuna
hyperparameter search), performs walk-forward backtesting, computes
evaluation metrics, optionally runs a betting simulation, generates an
HTML comparison report, and saves the best model(s) to the registry.

Usage
-----
::

    # Train all default models on v2 features
    python -m pipelines.train_and_evaluate

    # Only XGBoost and LightGBM with 100 Optuna trials
    python -m pipelines.train_and_evaluate \\
        --models xgboost lightgbm --n-trials 100

    # Include betting simulation
    python -m pipelines.train_and_evaluate --run-betting-sim

    # Use v4 features (includes odds)
    python -m pipelines.train_and_evaluate --feature-version v4
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
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config.settings import (
    FEATURES_DIR,
    N_OPTUNA_TRIALS,
    OUTPUTS_DIR,
    RANDOM_SEED,
    START_YEAR,
)
from evaluation.backtesting import WalkForwardBacktester
from evaluation.metrics import compare_models, compute_all_metrics
from evaluation.reports import generate_model_comparison_report
from modelling.classical_models import get_model, list_models as list_classical_models
from modelling.model_registry import save_model

logger = logging.getLogger(__name__)

# Default model list -- the classical models we always want to try.
_DEFAULT_CLASSICAL_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
]

# Baseline models that do not need feature-based training.
_BASELINE_NAMES = [
    "home_always",
    "ladder",
    "odds_implied",
    "elo",
]


# ============================================================================
# Data loading
# ============================================================================

def _load_features(feature_version: str, features_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the feature Parquet and extract the target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (features_df, target) -- the feature matrix and binary target.
    """
    path = features_dir / f"features_{feature_version}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}. "
            f"Run `python -m pipelines.build_features --feature-version {feature_version}` first."
        )

    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded features from %s: %d rows, %d cols.", path, len(df), len(df.columns))

    # Extract target
    if "target_home_win" in df.columns:
        target = df["target_home_win"].copy()
        df = df.drop(columns=["target_home_win"])
    else:
        raise ValueError(
            "Feature file does not contain 'target_home_win'. "
            "Rebuild features with `python -m pipelines.build_features`."
        )

    return df, target


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Identify numeric feature columns, excluding metadata columns."""
    meta_cols = {"year", "round", "season", "season_year", "date", "kickoff",
                 "home_team", "away_team", "venue", "match_id"}
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in meta_cols
    ]
    return feature_cols


# ============================================================================
# Baseline evaluation
# ============================================================================

def _run_baselines(
    features_df: pd.DataFrame,
    target: pd.Series,
    backtester: WalkForwardBacktester,
) -> dict[str, dict[str, Any]]:
    """Run baseline models through the backtester.

    Returns a dict of {model_name: metrics_dict} for each baseline.
    """
    from modelling.baseline_models import (
        HomeAlwaysModel,
        LadderModel,
        OddsImpliedModel,
        EloModel,
    )

    baselines: dict[str, tuple[Any, bool]] = {}

    # HomeAlways -- always available
    baselines["home_always"] = (lambda: HomeAlwaysModel(), False)

    # Ladder -- needs home_ladder_pos, away_ladder_pos
    if "home_ladder_pos" in features_df.columns and "away_ladder_pos" in features_df.columns:
        baselines["ladder"] = (lambda: LadderModel(), False)

    # OddsImplied -- needs home_odds / away_odds or similar
    has_odds_cols = any(
        c in features_df.columns for c in ["home_odds", "h2h_home", "home_open_prob"]
    )
    if has_odds_cols:
        baselines["odds_implied"] = (lambda: OddsImpliedModel(), False)

    # Elo -- needs home_team, away_team, season
    if all(c in features_df.columns for c in ["home_team", "away_team"]):
        baselines["elo"] = (lambda: EloModel(), False)

    results: dict[str, dict[str, Any]] = {}

    for name, (factory, needs_retraining) in baselines.items():
        print(f"\n    Running baseline: {name}...")
        try:
            per_year_df, preds_df = backtester.run(
                model_factory=factory,
                features_df=features_df,
                target=target,
                needs_retraining=needs_retraining,
            )

            if preds_df.empty:
                print(f"      -> No predictions produced. Skipping.")
                continue

            overall_metrics = compute_all_metrics(
                preds_df["y_true"], preds_df["y_pred"], preds_df["y_prob"]
            )
            overall_metrics["per_year"] = per_year_df
            results[name] = overall_metrics

            print(f"      Accuracy: {overall_metrics['accuracy']:.4f}  "
                  f"Log-Loss: {overall_metrics['log_loss']:.4f}")

        except Exception as exc:
            logger.error("Baseline '%s' failed: %s", name, exc)
            print(f"      -> FAILED: {exc}")

    return results


# ============================================================================
# Classical model training + evaluation
# ============================================================================

def _run_classical_models(
    model_names: list[str],
    features_df: pd.DataFrame,
    target: pd.Series,
    feature_cols: list[str],
    backtester: WalkForwardBacktester,
    n_trials: int = 0,
    save_best: bool = True,
    feature_version: str = "v2",
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], dict[str, Any]]:
    """Train, optionally tune, and backtest classical ML models.

    Returns
    -------
    tuple
        (all_results, predictions_by_model, fitted_models)
    """
    all_results: dict[str, dict[str, Any]] = {}
    predictions_by_model: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, Any] = {}

    for model_name in model_names:
        print(f"\n  {'='*50}")
        print(f"  Model: {model_name}")
        print(f"  {'='*50}")

        # --- Optional Optuna hyperparameter search ---
        best_params: dict[str, Any] | None = None
        if n_trials > 0:
            print(f"    Running Optuna search ({n_trials} trials)...")
            try:
                from modelling.hyperparameter_search import run_optuna_search

                # Use TimeSeriesSplit for Optuna CV (respects temporal ordering)
                cv = TimeSeriesSplit(n_splits=5)

                # Prepare training data (use all data before the last test year)
                train_mask = features_df["year"] < backtester.test_years[0]
                X_search = features_df.loc[train_mask, feature_cols]
                y_search = target.loc[train_mask]

                # Drop NaN targets
                valid = y_search.notna()
                X_search = X_search.loc[valid]
                y_search = y_search.loc[valid].astype(int)

                if len(X_search) < 100:
                    print(f"    -> Not enough training data for Optuna ({len(X_search)} rows). Skipping.")
                else:
                    best_params, study = run_optuna_search(
                        model_name=model_name,
                        X=X_search,
                        y=y_search.values,
                        cv_splitter=cv,
                        n_trials=n_trials,
                        metric="log_loss",
                    )
                    print(f"    Best log-loss: {study.best_value:.5f}")
                    print(f"    Best params: {best_params}")

            except Exception as exc:
                logger.error("Optuna search for '%s' failed: %s", model_name, exc)
                print(f"    -> Optuna FAILED: {exc}")
                best_params = None

        # --- Model factory (with or without tuned params) ---
        def make_model(params=best_params, name=model_name):
            return get_model(name, params)

        # --- Walk-forward backtest ---
        print(f"    Running walk-forward backtest...")
        t0 = time.time()
        try:
            # Create a features-only DataFrame for the backtester
            # (year + round + numeric feature columns)
            bt_cols = ["year", "round"] + feature_cols
            bt_cols = [c for c in bt_cols if c in features_df.columns]
            bt_df = features_df[bt_cols].copy()

            per_year_df, preds_df = backtester.run(
                model_factory=make_model,
                features_df=bt_df,
                target=target,
            )

            elapsed = time.time() - t0

            if preds_df.empty:
                print(f"    -> No predictions produced. Skipping.")
                continue

            # Compute aggregate metrics
            overall_metrics = compute_all_metrics(
                preds_df["y_true"], preds_df["y_pred"], preds_df["y_prob"]
            )
            overall_metrics["per_year"] = per_year_df
            if best_params is not None:
                overall_metrics["best_params"] = best_params

            all_results[model_name] = overall_metrics
            predictions_by_model[model_name] = preds_df

            print(f"    Accuracy : {overall_metrics['accuracy']:.4f}")
            print(f"    Log-Loss : {overall_metrics['log_loss']:.4f}")
            print(f"    Brier    : {overall_metrics.get('brier_score', float('nan')):.4f}")
            print(f"    AUC-ROC  : {overall_metrics.get('auc_roc', float('nan')):.4f}")
            print(f"    Time     : {elapsed:.1f}s")

            # --- Fit a final model on all available data (for saving) ---
            if save_best:
                print(f"    Fitting final model on full training data...")
                full_valid = target.notna()
                X_full = features_df.loc[full_valid, feature_cols]
                y_full = target.loc[full_valid].astype(int)

                final_model = make_model()
                final_model.fit(X_full, y_full)
                fitted_models[model_name] = final_model

        except Exception as exc:
            logger.error("Model '%s' failed: %s", model_name, exc)
            print(f"    -> FAILED: {exc}")

    return all_results, predictions_by_model, fitted_models


# ============================================================================
# Betting simulation
# ============================================================================

def _run_betting_simulation(
    predictions_by_model: dict[str, pd.DataFrame],
    features_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[int, dict[str, Any]]]]:
    """Run betting simulation for each model's predictions.

    Returns
    -------
    tuple
        (betting_by_model, betting_by_year)
    """
    from evaluation.betting_simulation import simulate_season

    betting_by_model: dict[str, dict[str, Any]] = {}
    betting_by_year: dict[str, dict[int, dict[str, Any]]] = {}

    # Find an odds column in the features
    odds_col = None
    for candidate in ["home_odds", "h2h_home", "home_open_prob"]:
        if candidate in features_df.columns:
            odds_col = candidate
            break

    if odds_col is None:
        print("    No odds column found in features. Skipping betting simulation.")
        return betting_by_model, betting_by_year

    for model_name, preds_df in predictions_by_model.items():
        print(f"\n    Betting sim: {model_name}...")

        try:
            # We need to align predictions with odds data.
            # preds_df has y_true, y_pred, y_prob, year columns.
            # We need to attach odds_decimal.
            if "odds_decimal" not in preds_df.columns:
                # Try to reconstruct odds from feature data
                # This is approximate -- the backtester only outputs y_true, y_pred, y_prob, year
                # For a proper simulation, we would need the original row indices.
                # For now, skip if we can't align the data.
                print(f"      -> Cannot align odds data with predictions. Skipping.")
                continue

            summary = simulate_season(preds_df, strategy="flat_value", threshold=0.05)
            betting_by_model[model_name] = summary

            print(f"      Bets placed : {summary['total_bets']}")
            print(f"      Win rate    : {summary['win_rate']:.1%}")
            print(f"      ROI         : {summary['roi']:+.1f}%")
            print(f"      P&L         : ${summary['profit_loss']:+.2f}")

            # Per-year breakdown
            if "year" in preds_df.columns:
                year_results: dict[int, dict[str, Any]] = {}
                for year, group in preds_df.groupby("year"):
                    if "odds_decimal" in group.columns:
                        yr_summary = simulate_season(group, strategy="flat_value")
                        year_results[int(year)] = yr_summary
                if year_results:
                    betting_by_year[model_name] = year_results

        except Exception as exc:
            logger.error("Betting sim for '%s' failed: %s", model_name, exc)
            print(f"      -> FAILED: {exc}")

    return betting_by_model, betting_by_year


# ============================================================================
# Core pipeline
# ============================================================================

def train_and_evaluate(
    model_names: list[str] | None = None,
    feature_version: str = "v2",
    n_trials: int = 0,
    run_betting_sim: bool = False,
    save_best: bool = True,
    test_years: list[int] | None = None,
    features_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute the full training and evaluation pipeline.

    Parameters
    ----------
    model_names : list[str], optional
        Classical model names to train.  If None, uses the default set:
        logistic_regression, random_forest, xgboost, lightgbm.
    feature_version : str
        Feature set version (v1-v4).
    n_trials : int
        Number of Optuna trials.  0 means no hyperparameter search.
    run_betting_sim : bool
        Whether to run the betting simulation.
    save_best : bool
        Whether to save the best model(s) to the registry.
    test_years : list[int], optional
        Years to use as test folds.  Defaults to 2018-2025.
    features_dir : Path, optional
        Directory to load feature Parquet from.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: ``"baseline_results"``, ``"model_results"``,
        ``"predictions_by_model"``, ``"fitted_models"``, ``"comparison_df"``,
        and optionally ``"betting_results"``.
    """
    if model_names is None:
        model_names = list(_DEFAULT_CLASSICAL_MODELS)
    if test_years is None:
        test_years = list(range(2018, 2026))
    if features_dir is None:
        features_dir = FEATURES_DIR

    # ------------------------------------------------------------------
    # 1. Load features
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  [1/5] Loading feature matrix ({feature_version})")
    print(f"{'='*60}")

    features_df, target = _load_features(feature_version, features_dir)
    feature_cols = _get_feature_columns(features_df)

    # Drop rows with NaN target for training/evaluation
    valid_mask = target.notna()
    n_valid = valid_mask.sum()
    n_total = len(target)
    print(f"    Total rows: {n_total:,}  |  Valid targets: {n_valid:,}  |  NaN: {n_total - n_valid:,}")
    print(f"    Feature columns: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # 2. Set up backtester
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  [2/5] Setting up walk-forward backtester")
    print(f"{'='*60}")

    backtester = WalkForwardBacktester(
        train_start_year=START_YEAR,
        test_years=test_years,
        expanding=True,
        year_column="year",
        round_column="round",
    )

    available_years = features_df["year"].dropna().unique()
    print(f"    Train start year : {START_YEAR}")
    print(f"    Test years       : {test_years[0]} - {test_years[-1]}")
    print(f"    Available years  : {sorted(available_years.astype(int))}")
    print(f"    Window mode      : Expanding")

    # ------------------------------------------------------------------
    # 3. Run baselines
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  [3/5] Running baseline models")
    print(f"{'='*60}")

    # Baselines need the full DataFrame (with non-numeric columns like
    # home_team, away_team, etc.) because they use named columns directly.
    baseline_results = _run_baselines(features_df, target, backtester)

    # ------------------------------------------------------------------
    # 4. Train and backtest classical models
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  [4/5] Training classical models")
    print(f"{'='*60}")
    print(f"    Models    : {model_names}")
    print(f"    Optuna    : {'Yes (' + str(n_trials) + ' trials)' if n_trials > 0 else 'No'}")

    model_results, predictions_by_model, fitted_models = _run_classical_models(
        model_names=model_names,
        features_df=features_df,
        target=target,
        feature_cols=feature_cols,
        backtester=backtester,
        n_trials=n_trials,
        save_best=save_best,
        feature_version=feature_version,
    )

    # ------------------------------------------------------------------
    # 5. Combine results, generate report, save models
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  [5/5] Generating report and saving models")
    print(f"{'='*60}")

    # Merge baseline + classical results
    all_results = {**baseline_results, **model_results}

    # Comparison table
    metrics_for_comparison = {
        name: {k: v for k, v in vals.items() if isinstance(v, (int, float))}
        for name, vals in all_results.items()
    }
    comparison_df = compare_models(metrics_for_comparison)

    print(f"\n  Model Comparison (sorted by accuracy):")
    print(f"  {'-'*70}")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.max_columns", 10):
        print(comparison_df.to_string())
    print()

    # Betting simulation (optional)
    betting_by_model: dict[str, dict[str, Any]] = {}
    betting_by_year: dict[str, dict[int, dict[str, Any]]] = {}

    if run_betting_sim:
        print("  Running betting simulation...")
        betting_by_model, betting_by_year = _run_betting_simulation(
            predictions_by_model, features_df,
        )

    # Generate HTML report
    print("  Generating HTML report...")
    try:
        report_html = generate_model_comparison_report(
            all_results=all_results,
            predictions_by_model=predictions_by_model,
            models=fitted_models if fitted_models else None,
            feature_names=feature_cols if fitted_models else None,
            betting_results_by_model=betting_by_model if betting_by_model else None,
            betting_results_by_year=betting_by_year if betting_by_year else None,
            save=True,
            filename=f"model_comparison_{feature_version}.html",
        )
        print(f"    Report saved to: {OUTPUTS_DIR / 'reports' / f'model_comparison_{feature_version}.html'}")
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        print(f"    -> Report generation FAILED: {exc}")

    # Save best model(s) to registry
    if save_best and fitted_models:
        print("\n  Saving models to registry...")

        # Determine the best model by log-loss (lower is better)
        best_name = None
        best_log_loss = float("inf")
        for name, metrics in model_results.items():
            ll = metrics.get("log_loss", float("inf"))
            if ll < best_log_loss:
                best_log_loss = ll
                best_name = name

        for name, model in fitted_models.items():
            is_best = (name == best_name)
            metadata = {
                "feature_version": feature_version,
                "feature_columns": feature_cols,
                "n_features": len(feature_cols),
                "is_best": is_best,
                **{k: v for k, v in model_results.get(name, {}).items()
                   if isinstance(v, (int, float, str, bool))},
            }
            if n_trials > 0 and "best_params" in model_results.get(name, {}):
                metadata["best_params"] = model_results[name]["best_params"]

            registry_name = f"{name}_{feature_version}"
            saved_path = save_model(model, name=registry_name, metadata=metadata)
            tag = " [BEST]" if is_best else ""
            print(f"    Saved: {registry_name}{tag} -> {saved_path}")

    # Build return dictionary
    pipeline_results: dict[str, Any] = {
        "baseline_results": baseline_results,
        "model_results": model_results,
        "predictions_by_model": predictions_by_model,
        "fitted_models": fitted_models,
        "comparison_df": comparison_df,
        "all_results": all_results,
    }
    if run_betting_sim:
        pipeline_results["betting_by_model"] = betting_by_model
        pipeline_results["betting_by_year"] = betting_by_year

    return pipeline_results


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="pipelines.train_and_evaluate",
        description="Train and evaluate NRL match winner prediction models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Classical model names to train. "
            f"Default: {_DEFAULT_CLASSICAL_MODELS}. "
            f"Available: {list_classical_models()}"
        ),
    )
    parser.add_argument(
        "--feature-version",
        type=str,
        default="v2",
        choices=["v1", "v2", "v3", "v4"],
        help="Feature set version (default: v2).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=0,
        help=(
            "Number of Optuna hyperparameter search trials per model. "
            f"0 = skip search, use defaults. Typical: 50-200."
        ),
    )
    parser.add_argument(
        "--run-betting-sim",
        action="store_true",
        default=False,
        help="Run betting simulation after backtesting.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Do not save trained models to the registry.",
    )
    parser.add_argument(
        "--test-years",
        nargs="+",
        type=int,
        default=None,
        help="Years to use as test folds (default: 2018-2025).",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help=f"Directory containing feature Parquet files (default: {FEATURES_DIR}).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the training and evaluation pipeline."""
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\nNRL Model Training & Evaluation Pipeline")
    print(f"  Feature version : {args.feature_version}")
    print(f"  Models          : {args.models or _DEFAULT_CLASSICAL_MODELS}")
    print(f"  Optuna trials   : {args.n_trials}")
    print(f"  Betting sim     : {'Yes' if args.run_betting_sim else 'No'}")
    print(f"  Save models     : {'No' if args.no_save else 'Yes'}")

    t_start = time.time()

    results = train_and_evaluate(
        model_names=args.models,
        feature_version=args.feature_version,
        n_trials=args.n_trials,
        run_betting_sim=args.run_betting_sim,
        save_best=not args.no_save,
        test_years=args.test_years,
        features_dir=Path(args.features_dir) if args.features_dir else None,
    )

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed time: {elapsed:.1f}s")
    print("  Done.\n")


if __name__ == "__main__":
    main()
