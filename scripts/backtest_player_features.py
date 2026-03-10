#!/usr/bin/env python3
"""
Backtest: Player-Level Form Features
=====================================
Validates whether per-player rolling form stats improve NRL match prediction
accuracy over the current V4 baseline features.

Walk-forward design (no look-ahead bias):
  For each test season Y in [2023, 2024, 2025]:
    - Train on all seasons before Y (with exponential recency weighting)
    - Test on season Y
    - Compare Baseline (V4 top-60) vs Enhanced (V4 + player form top-60)

Player form features (aggregated to team level per match):
  Rolling 3/5 game averages per player → team-level aggregates:
  - player_form_score_3/5     — composite attack score per starter
  - spine_form_score_3/5      — spine players (FB/HB/FE/HK) form
  - player_run_metres_3/5     — rolling run metres per starter
  - player_tackles_3/5        — rolling tackles per starter
  - player_miss_tackle_rate_3/5 — team's rolling missed tackle rate
  - player_errors_3/5         — rolling errors per starter
  (home / away / diff variants → ~36 new features total)

Graceful degradation:
  If player_match_stats.parquet is missing, prints a warning and runs
  baseline-only mode (no Enhanced column).

Usage:
    python scripts/backtest_player_features.py
    python scripts/backtest_player_features.py --test-years 2024 2025
    python scripts/backtest_player_features.py --test-years 2023 2024 2025 --min-train-year 2015
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import accuracy_score, log_loss
from catboost import CatBoostClassifier
import xgboost as xgb

from config.settings import PROCESSED_DIR
from pipelines import v3, v4
from predict_round import (
    BEST_CAT_PARAMS,
    SAMPLE_WEIGHT_DECAY,
    FEATURE_COLS,
    load_historical_data,
    get_elo_params,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAYER_STATS_PATH = PROCESSED_DIR / "player_match_stats.parquet"

# Stats to roll per player — camelCase (NRL API) and snake_case fallbacks
# are both handled by _detect_stat_columns()
ATTACK_STAT_KEYWORDS = ["linebreak", "runmetre", "run_metre", "allrun", "all_run",
                         "offload", "tacklebreak", "tackle_break"]
DEFENCE_STAT_KEYWORDS = ["tackle", "missed", "error"]

ROLLING_WINDOWS = [3, 5]

FEATURE_SUFFIXES = [
    "form_score",
    "spine_form",
    "run_metres",
    "tackles",
    "miss_tackle_rate",
    "errors",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_player_match_stats() -> pd.DataFrame | None:
    """Load per-player match stats. Returns None if file doesn't exist."""
    if not PLAYER_STATS_PATH.exists():
        print("\n  ⚠  WARNING: player_match_stats.parquet not found.")
        print("     To generate it, run the Monday refresh pipeline:")
        print("       python refresh_week.py           # adds this round's player stats")
        print("     Or to backfill historical seasons, see scraping/nrl_match_stats.py")
        print("  Proceeding with BASELINE ONLY.\n")
        return None

    df = pd.read_parquet(PLAYER_STATS_PATH)
    n_seasons = df["year"].nunique() if "year" in df.columns else "?"
    n_players = df["player_id"].nunique() if "player_id" in df.columns else "?"
    print(f"  Loaded {len(df):,} player-match rows "
          f"({n_seasons} seasons, {n_players} unique players)")
    return df


# ---------------------------------------------------------------------------
# Full V4 feature pipeline
# ---------------------------------------------------------------------------

def build_all_v4_features(
    matches: pd.DataFrame,
    ladders: pd.DataFrame,
    odds: pd.DataFrame,
    match_stats: pd.DataFrame | None,
    elo_params: dict,
) -> pd.DataFrame:
    """Run the full V3+V4 feature pipeline on historical matches only.

    Mirrors the feature build in predict_round.build_features() but without
    the upcoming-match split.  Draws are excluded (home_score == away_score).
    """
    linked = v3.link_odds(matches, odds)
    linked = linked.dropna(subset=["home_score"]).reset_index(drop=True)
    linked = linked[linked["home_score"] != linked["away_score"]].reset_index(drop=True)

    all_m = linked.sort_values("date").reset_index(drop=True)

    # V3 base pipeline
    all_m = v3.backfill_elo(all_m, elo_params)
    all_m = v3.compute_rolling_form_features(all_m)
    all_m = v3.compute_h2h_features(all_m)
    all_m = v3.compute_ladder_features(all_m, ladders)
    all_m = v3.compute_venue_features(all_m)
    all_m = v3.compute_odds_features(all_m)
    all_m = v3.compute_schedule_features(all_m)
    all_m = v3.compute_contextual_features(all_m)
    all_m = v3.compute_engineered_features(all_m)

    # V4 extended pipeline
    all_m = v4.compute_v4_odds_features(all_m)
    all_m = v4.compute_scoring_consistency_features(all_m)
    all_m = v4.compute_attendance_features(all_m)
    all_m = v4.compute_kickoff_features(all_m)
    all_m = v4.compute_lineup_stability_features(all_m)
    all_m = v4.compute_player_impact_features(all_m)
    all_m = v4.compute_team_stats_features(all_m)
    all_m = v4.compute_referee_features(all_m)
    all_m = v4.compute_v4_engineered_features(all_m)
    all_m = v4.compute_rolling_match_stats_features(all_m, match_stats)

    # Target
    all_m["home_win"] = np.where(
        all_m["home_score"] > all_m["away_score"], 1.0,
        np.where(all_m["home_score"] < all_m["away_score"], 0.0, np.nan),
    )
    all_m = all_m.dropna(subset=["home_win"]).reset_index(drop=True)

    return all_m


# ---------------------------------------------------------------------------
# Player form feature computation
# ---------------------------------------------------------------------------

def _to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _detect_stat_columns(ps: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Auto-detect attack, defence, and run-metres stat columns.

    Returns (attack_cols, defence_cols, all_stat_cols).
    """
    # All numeric columns that aren't metadata
    META = {"year", "round", "match_id", "player_id", "player_name",
            "team", "home_team", "away_team", "is_spine", "is_starter",
            "position", "jersey_number", "minutes_played", "minutesPlayed"}
    numeric = [
        c for c in ps.columns
        if c not in META and pd.api.types.is_numeric_dtype(ps[c])
    ]
    attack = [c for c in numeric
              if any(kw in c.lower() for kw in ATTACK_STAT_KEYWORDS)]
    defence = [c for c in numeric
               if any(kw in c.lower() for kw in DEFENCE_STAT_KEYWORDS)]
    # Remove duplicates (some cols match both)
    seen: set[str] = set()
    all_cols: list[str] = []
    for c in attack + defence:
        if c not in seen:
            seen.add(c)
            all_cols.append(c)
    return attack, defence, all_cols


def _round_to_int(r) -> int:
    """Convert round value to sortable integer (finals → 100+)."""
    try:
        return int(r)
    except (ValueError, TypeError):
        rs = str(r).lower()
        if "qualif" in rs:
            return 100
        if "elim" in rs:
            return 101
        if "semi" in rs:
            return 102
        if "prelim" in rs:
            return 103
        if "grand" in rs:
            return 104
        return 99


def compute_player_form_features(
    matches_df: pd.DataFrame,
    player_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute team-level player form features from raw per-player match stats.

    For each match, aggregates rolling per-player stats (over the last 3 and 5
    matches) across starters into team-level features.  No look-ahead bias:
    rolling windows only reference matches with an index BEFORE the current one.

    Added columns (× home/away/diff = ×3):
      {side}_player_form_score_{w}      composite attack score per starter
      {side}_player_spine_form_{w}      spine (FB/HB/FE/HK) attack form
      {side}_player_run_metres_{w}      rolling run metres per starter
      {side}_player_tackles_{w}         rolling tackles per starter
      {side}_player_miss_tackle_rate_{w} rolling missed tackle rate
      {side}_player_errors_{w}          rolling errors per starter
      player_{suffix}_diff_{w}          home minus away differential

    Parameters
    ----------
    matches_df : pd.DataFrame
        Full historical matches DataFrame (already sorted chronologically).
    player_stats_df : pd.DataFrame
        Raw per-player per-match stats from player_match_stats.parquet.
        Expected columns: year, round, team, player_id / player_name,
        plus numeric stat columns (camelCase or snake_case).

    Returns
    -------
    pd.DataFrame
        matches_df with new player form feature columns appended.
    """
    print("\n" + "=" * 80)
    print("  COMPUTING PLAYER FORM FEATURES")
    print("=" * 80)

    ps = player_stats_df.copy()
    df = matches_df.copy().reset_index(drop=True)
    df["_match_idx"] = range(len(df))

    # ── Detect available stat columns ─────────────────────────────────────
    attack_cols, defence_cols, all_stat_cols = _detect_stat_columns(ps)
    if not all_stat_cols:
        print("  WARNING: No numeric stat columns found in player_match_stats — skipping")
        return df.drop(columns=["_match_idx"], errors="ignore")

    print(f"  Attack stats  : {attack_cols}")
    print(f"  Defence stats : {defence_cols}")

    # ── Normalise key join columns ─────────────────────────────────────────
    if "year" not in ps.columns:
        print("  WARNING: player_stats_df missing 'year' column — skipping")
        return df.drop(columns=["_match_idx"], errors="ignore")

    ps["year"] = pd.to_numeric(ps["year"], errors="coerce").astype("Int64")
    ps["_round_num"] = ps["round"].apply(_round_to_int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["_round_num"] = df["round"].apply(_round_to_int)

    # ── Join player stats rows to the matches DataFrame ──────────────────
    # This gives each player-stat row its chronological match_idx + date,
    # which is required for no-look-ahead rolling.
    if "home_team" in ps.columns and "away_team" in ps.columns:
        join_keys = ["year", "_round_num", "home_team", "away_team"]
        merge_ref = df[["_match_idx", "year", "_round_num",
                        "home_team", "away_team", "date"]].copy()
        ps_merged = ps.merge(merge_ref, on=join_keys, how="inner")
    else:
        # Fallback: join only on year + round (less precise, may create duplicates)
        print("  INFO: player_stats lacks home_team/away_team — joining on year+round only")
        merge_ref = df[["_match_idx", "year", "_round_num", "date"]].copy()
        ps_merged = ps.merge(merge_ref, on=["year", "_round_num"], how="inner")

    if ps_merged.empty:
        print("  WARNING: No player stats joined to matches — "
              "check team name standardisation")
        return df.drop(columns=["_match_idx", "_round_num"], errors="ignore")

    print(f"  Joined {len(ps_merged):,} player-match rows to {len(df)} matches")

    # ── Ensure stat columns are numeric ───────────────────────────────────
    for col in all_stat_cols:
        ps_merged[col] = pd.to_numeric(ps_merged[col], errors="coerce")

    # ── Detect is_spine flag ──────────────────────────────────────────────
    SPINE_POS = {"FB", "HB", "FE", "HK"}
    if "is_spine" in ps_merged.columns:
        ps_merged["is_spine"] = ps_merged["is_spine"].fillna(False).astype(bool)
    elif "position" in ps_merged.columns:
        ps_merged["is_spine"] = ps_merged["position"].str.upper().str.strip().isin(SPINE_POS)
    else:
        ps_merged["is_spine"] = False  # no spine enrichment without position data

    # ── Filter to starters (if available) ────────────────────────────────
    if "is_starter" in ps_merged.columns:
        starters = ps_merged[ps_merged["is_starter"].fillna(True).astype(bool)].copy()
    else:
        starters = ps_merged.copy()

    # ── Identify player key ───────────────────────────────────────────────
    player_id_col = "player_id" if "player_id" in starters.columns else "player_name"

    starters = starters.sort_values(
        [player_id_col, "date", "_match_idx"]
    ).reset_index(drop=True)

    # ── Find specific columns for aggregation ─────────────────────────────
    run_col = next((c for c in attack_cols
                    if "runmetre" in c.lower() or "run_metre" in c.lower()), None)
    tackle_col = next((c for c in defence_cols
                       if "tackle" in c.lower() and "missed" not in c.lower()
                       and "break" not in c.lower()), None)
    miss_col = next((c for c in defence_cols if "missed" in c.lower()), None)
    err_col = next((c for c in defence_cols if "error" in c.lower()), None)

    print(f"  Key cols  →  run: {run_col}, tackle: {tackle_col}, "
          f"missed: {miss_col}, error: {err_col}")

    # ── Build per-team rolling lookup ─────────────────────────────────────
    # lookup[(team, match_idx)] = {f"{suffix}_{window}": value, ...}
    lookup: dict[tuple[str, int], dict[str, float]] = {}

    for team in starters["team"].unique():
        team_df = starters[starters["team"] == team].copy()
        # Sorted list of match_idxs this team appeared in
        team_midxs = sorted(team_df["_match_idx"].unique())

        for i, midx in enumerate(team_midxs):
            key = (team, midx)
            lookup[key] = {}

            for w in ROLLING_WINDOWS:
                # Prior match indices only (strict look-back, no current match)
                prior_idxs = team_midxs[max(0, i - w): i]
                if not prior_idxs:
                    for suf in FEATURE_SUFFIXES:
                        lookup[key][f"{suf}_{w}"] = np.nan
                    continue

                prior = team_df[team_df["_match_idx"].isin(prior_idxs)]
                spine_prior = prior[prior["is_spine"]]

                # ── Attack composite (mean of attack stat means per match) ──
                match_attack_means = []
                for stat in attack_cols:
                    pm = prior.groupby("_match_idx")[stat].mean()
                    match_attack_means.append(pm.mean())
                form_score = float(np.nanmean(match_attack_means)) if match_attack_means else np.nan

                # ── Spine attack composite ─────────────────────────────────
                spine_vals = []
                for stat in attack_cols[:2]:  # lineBreaks + runMetres are most signal-rich
                    if len(spine_prior) > 0 and stat in spine_prior.columns:
                        pm = spine_prior.groupby("_match_idx")[stat].mean()
                        spine_vals.append(pm.mean())
                spine_score = float(np.nanmean(spine_vals)) if spine_vals else np.nan

                # ── Run metres per starter ──────────────────────────────────
                avg_run = np.nan
                if run_col and run_col in prior.columns:
                    avg_run = float(prior.groupby("_match_idx")[run_col].mean().mean())

                # ── Tackles per starter ────────────────────────────────────
                avg_tackles = np.nan
                if tackle_col and tackle_col in prior.columns:
                    avg_tackles = float(prior.groupby("_match_idx")[tackle_col].mean().mean())

                # ── Missed tackle rate per match ───────────────────────────
                miss_rate = np.nan
                if miss_col and miss_col in prior.columns:
                    pm_miss = prior.groupby("_match_idx")[miss_col].sum()
                    if tackle_col and tackle_col in prior.columns:
                        pm_tackle = prior.groupby("_match_idx")[tackle_col].sum()
                        denom = (pm_tackle + pm_miss).replace(0, np.nan)
                        miss_rate = float((pm_miss / denom).mean())
                    else:
                        miss_rate = float(pm_miss.mean())

                # ── Errors per starter ─────────────────────────────────────
                avg_errors = np.nan
                if err_col and err_col in prior.columns:
                    avg_errors = float(prior.groupby("_match_idx")[err_col].mean().mean())

                lookup[key][f"form_score_{w}"] = form_score
                lookup[key][f"spine_form_{w}"] = spine_score
                lookup[key][f"run_metres_{w}"] = avg_run
                lookup[key][f"tackles_{w}"] = avg_tackles
                lookup[key][f"miss_tackle_rate_{w}"] = miss_rate
                lookup[key][f"errors_{w}"] = avg_errors

    n_keys = len(lookup)
    n_valid = sum(
        1 for v in lookup.values() if not np.isnan(v.get("form_score_3", np.nan))
    )
    print(f"  Built rolling lookup: {n_keys} (team, match) keys | "
          f"{n_valid} with valid 3-game form score")

    # ── Attach features to the matches DataFrame ──────────────────────────
    n_added = 0
    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        for w in ROLLING_WINDOWS:
            for suffix in FEATURE_SUFFIXES:
                col_name = f"{side}_player_{suffix}_{w}"
                df[col_name] = [
                    lookup.get((df.at[i, team_col], i), {}).get(f"{suffix}_{w}", np.nan)
                    for i in range(len(df))
                ]
                n_added += 1

    # Home − away differentials
    for w in ROLLING_WINDOWS:
        for suffix in FEATURE_SUFFIXES:
            h_col = f"home_player_{suffix}_{w}"
            a_col = f"away_player_{suffix}_{w}"
            d_col = f"player_{suffix}_diff_{w}"
            df[d_col] = df[h_col] - df[a_col]
            n_added += 1

    # Coverage check
    first_col = "home_player_form_score_3"
    if first_col in df.columns:
        cov = df[first_col].notna().mean() * 100
        print(f"  Added {n_added} player form features | Coverage: {cov:.0f}%")
    else:
        print(f"  Added {n_added} player form features")

    df = df.drop(columns=["_match_idx", "_round_num"], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def select_top_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    feature_cols: list[str],
    top_n: int = 60,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Select top N features by XGBoost importance (matches predict_round.py)."""
    selector = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.02,
        verbosity=0, random_state=42,
    )
    selector.fit(X_train[feature_cols], y_train, sample_weight=sample_weights)
    imp = pd.Series(selector.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top = list(imp.head(top_n).index)
    ranked = list(imp.items())
    return top, ranked


def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    feature_cols: list[str],
) -> CatBoostClassifier:
    """Train CatBoost classifier on selected features."""
    params = dict(BEST_CAT_PARAMS)
    params["verbose"] = 0
    model = CatBoostClassifier(**params)
    model.fit(X_train[feature_cols], y_train, sample_weight=sample_weights)
    return model


def fill_missing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill NaN values with training medians (consistent with predict_round.py)."""
    BOOL_COLS = {
        "home_is_back_to_back", "away_is_back_to_back",
        "home_bye_last_round", "away_bye_last_round",
        "is_finals", "odds_home_favourite", "is_home", "is_neutral_venue",
        "odds_spread_agree", "elo_spread_agree",
        "attendance_high", "is_night_game", "is_afternoon_game", "is_day_game",
        "is_early_season", "is_mid_season", "is_late_season",
    }
    medians = train_df[cols].median()
    Xtr = train_df[cols].copy()
    Xte = test_df[cols].copy()
    for col in cols:
        fill_val = 0 if col in BOOL_COLS else medians.get(col, 0)
        if pd.isna(fill_val):
            fill_val = 0
        Xtr[col] = Xtr[col].fillna(fill_val)
        Xte[col] = Xte[col].fillna(fill_val)
    return Xtr, Xte


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Compute accuracy and log loss."""
    preds = (probs >= 0.5).astype(float)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "log_loss": float(log_loss(y_true, np.clip(probs, 1e-7, 1 - 1e-7))),
    }


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def run_backtest(
    all_data: pd.DataFrame,
    baseline_cols: list[str],
    player_cols: list[str],
    test_years: list[int],
    min_train_year: int = 2014,
    top_n: int = 60,
) -> dict:
    """Walk-forward comparison: baseline vs player-enhanced model.

    For each test year:
      - Train set: all data from min_train_year to test_year - 1
      - Test set: test_year
      - Baseline: CatBoost on top-{top_n} V4 features
      - Enhanced: CatBoost on top-{top_n} V4 + player form features
      - Odds: bookmaker implied probability (benchmark)

    Returns dict with 'results' DataFrame and 'importance' DataFrame.
    """
    print("\n" + "=" * 80)
    print(f"  WALK-FORWARD BACKTEST  (top-{top_n} feature selection)")
    print("=" * 80)

    has_player = len(player_cols) > 0
    combined_cols = baseline_cols + [c for c in player_cols if c not in baseline_cols]

    results: list[dict] = []
    importance_records: list[dict] = []

    for test_year in test_years:
        t0 = time.time()

        train_mask = (
            (all_data["year"] >= min_train_year) &
            (all_data["year"] < test_year)
        )
        test_mask = all_data["year"] == test_year

        train_df = all_data[train_mask].copy()
        test_df = all_data[test_mask].copy()

        if len(train_df) < 100:
            print(f"\n  {test_year}: SKIPPED — insufficient training data ({len(train_df)} rows)")
            continue
        if len(test_df) < 5:
            print(f"\n  {test_year}: SKIPPED — insufficient test data ({len(test_df)} rows)")
            continue

        y_train = train_df["home_win"].values.astype(float)
        y_test = test_df["home_win"].values.astype(float)

        train_yrs = train_df["year"].values
        sample_weights = SAMPLE_WEIGHT_DECAY ** (train_yrs.max() - train_yrs)

        # Odds benchmark
        odds_probs = test_df.get("odds_home_prob", pd.Series(np.full(len(test_df), 0.55))).values
        odds_probs = np.where(np.isnan(odds_probs), 0.55, odds_probs)
        odds_probs = np.clip(odds_probs, 1e-7, 1 - 1e-7)
        odds_metrics = compute_metrics(y_test, odds_probs)

        # ── BASELINE: top-N from V4 features ─────────────────────────────
        Xtr_b, Xte_b = fill_missing(train_df, test_df, baseline_cols)
        top_base, ranked_base = select_top_features(
            Xtr_b, y_train, sample_weights, baseline_cols, top_n=top_n
        )
        model_base = train_catboost(Xtr_b, y_train, sample_weights, top_base)
        probs_base = np.clip(
            model_base.predict_proba(Xte_b[top_base])[:, 1], 1e-7, 1 - 1e-7
        )
        metrics_base = compute_metrics(y_test, probs_base)

        # ── ENHANCED: top-N from V4 + player form features ───────────────
        metrics_enh: dict | None = None
        top_enh: list[str] = []
        ranked_enh: list[tuple[str, float]] = []
        player_in_top = 0

        if has_player:
            Xtr_e, Xte_e = fill_missing(train_df, test_df, combined_cols)
            top_enh, ranked_enh = select_top_features(
                Xtr_e, y_train, sample_weights, combined_cols, top_n=top_n
            )
            model_enh = train_catboost(Xtr_e, y_train, sample_weights, top_enh)
            probs_enh = np.clip(
                model_enh.predict_proba(Xte_e[top_enh])[:, 1], 1e-7, 1 - 1e-7
            )
            metrics_enh = compute_metrics(y_test, probs_enh)

            player_feats_in_top = [c for c in top_enh if c in player_cols]
            player_in_top = len(player_feats_in_top)

            for rank, (feat, imp_val) in enumerate(ranked_enh, start=1):
                importance_records.append({
                    "year": test_year,
                    "feature": feat,
                    "importance": imp_val,
                    "rank": rank,
                    "is_player_feature": feat in player_cols,
                })

        elapsed = time.time() - t0
        n_test = int(len(test_df))

        # ── Print fold summary ────────────────────────────────────────────
        print(f"\n  {test_year}  ({n_test} games, {elapsed:.0f}s)")
        print(f"    {'Model':<32s}  {'Accuracy':>9s}  {'Log Loss':>9s}  {'vs Odds':>8s}")
        print(f"    {'─' * 62}")

        def _vs(acc: float) -> str:
            d = (acc - odds_metrics["accuracy"]) * 100
            return f"{'+' if d >= 0 else ''}{d:.1f}pp"

        print(f"    {'Odds (implied)':<32s}  "
              f"{odds_metrics['accuracy']:>8.1%}  {odds_metrics['log_loss']:>9.4f}  {'—':>8s}")
        print(f"    {'Baseline  (V4 top-' + str(top_n) + ')':<32s}  "
              f"{metrics_base['accuracy']:>8.1%}  {metrics_base['log_loss']:>9.4f}  "
              f"{_vs(metrics_base['accuracy']):>8s}")

        if metrics_enh is not None:
            print(f"    {'Enhanced  (+player top-' + str(top_n) + ')':<32s}  "
                  f"{metrics_enh['accuracy']:>8.1%}  {metrics_enh['log_loss']:>9.4f}  "
                  f"{_vs(metrics_enh['accuracy']):>8s}")

            print(f"\n    Player features in top-{top_n}: {player_in_top}")
            player_feats_in_top = [c for c in top_enh if c in player_cols]
            if player_feats_in_top:
                print(f"    Top player features:")
                for feat in player_feats_in_top[:10]:
                    feat_rank = next(
                        (r for r, (f, _) in enumerate(ranked_enh, 1) if f == feat), "?"
                    )
                    feat_imp = next((v for f, v in ranked_enh if f == feat), 0.0)
                    print(f"      #{feat_rank:<3}  {feat:<50s}  imp={feat_imp:.1f}")

        results.append({
            "year": test_year,
            "n_games": n_test,
            "odds_accuracy": odds_metrics["accuracy"],
            "odds_log_loss": odds_metrics["log_loss"],
            "baseline_accuracy": metrics_base["accuracy"],
            "baseline_log_loss": metrics_base["log_loss"],
            "enhanced_accuracy": metrics_enh["accuracy"] if metrics_enh else None,
            "enhanced_log_loss": metrics_enh["log_loss"] if metrics_enh else None,
            "player_in_top": player_in_top,
        })

    return {
        "results": pd.DataFrame(results) if results else pd.DataFrame(),
        "importance": (
            pd.DataFrame(importance_records)
            if importance_records
            else pd.DataFrame()
        ),
    }


# ---------------------------------------------------------------------------
# Output / reporting
# ---------------------------------------------------------------------------

def print_comparison_table(results_df: pd.DataFrame, has_player: bool) -> None:
    """Print a clear side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("  SUMMARY: Baseline vs Player-Enhanced Features")
    print("=" * 80)

    if results_df.empty:
        print("  No results to display.")
        return

    total_games = int(results_df["n_games"].sum())
    weights = results_df["n_games"].values

    avg_odds = float(np.average(results_df["odds_accuracy"], weights=weights))
    avg_base_acc = float(np.average(results_df["baseline_accuracy"], weights=weights))
    avg_base_ll = float(np.average(results_df["baseline_log_loss"], weights=weights))

    # Column header
    hdr = (f"  {'Season':<8s}  {'N':>5s}  {'Odds Acc':>9s}  "
           f"{'Baseline Acc':>12s}  {'Baseline LL':>11s}")
    if has_player:
        hdr += f"  {'Enhanced Acc':>12s}  {'Enhanced LL':>11s}  {'Δ Acc':>7s}  {'Player/top60':>12s}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for _, row in results_df.iterrows():
        line = (f"  {int(row['year']):<8d}  {int(row['n_games']):>5d}  "
                f"{row['odds_accuracy']:>8.1%}  {row['baseline_accuracy']:>11.1%}  "
                f"{row['baseline_log_loss']:>11.4f}")
        if has_player and row["enhanced_accuracy"] is not None:
            delta = (row["enhanced_accuracy"] - row["baseline_accuracy"]) * 100
            line += (f"  {row['enhanced_accuracy']:>11.1%}  "
                     f"{row['enhanced_log_loss']:>11.4f}  "
                     f"{delta:>+6.1f}pp  "
                     f"{int(row['player_in_top']):>10d}/60")
        print(line)

    # Totals row
    print("  " + "─" * (len(hdr) - 2))
    summary = (f"  {'OVERALL':<8s}  {total_games:>5d}  "
               f"{avg_odds:>8.1%}  {avg_base_acc:>11.1%}  {avg_base_ll:>11.4f}")
    if has_player and results_df["enhanced_accuracy"].notna().any():
        valid = results_df["enhanced_accuracy"].notna()
        avg_enh_acc = float(np.average(
            results_df.loc[valid, "enhanced_accuracy"],
            weights=results_df.loc[valid, "n_games"],
        ))
        avg_enh_ll = float(np.average(
            results_df.loc[valid, "enhanced_log_loss"],
            weights=results_df.loc[valid, "n_games"],
        ))
        delta_overall = (avg_enh_acc - avg_base_acc) * 100
        avg_p60 = results_df["player_in_top"].mean()
        summary += (f"  {avg_enh_acc:>11.1%}  {avg_enh_ll:>11.4f}  "
                    f"{delta_overall:>+6.1f}pp  {avg_p60:>9.1f}/60")
    print(summary)

    # Conclusion
    print()
    if not has_player:
        print("  ℹ  No player_match_stats.parquet found — baseline only.")
        print("     Generate player stats by running the Monday refresh pipeline.")
    else:
        valid = results_df["enhanced_accuracy"].notna()
        if not valid.any():
            print("  ℹ  Enhanced model had no valid results to compare.")
            return
        avg_enh_acc_raw = results_df.loc[valid, "enhanced_accuracy"].mean()
        avg_base_raw = results_df.loc[valid, "baseline_accuracy"].mean()
        diff = (avg_enh_acc_raw - avg_base_raw) * 100
        avg_p = results_df["player_in_top"].mean()

        if diff > 0.5:
            print(f"  ✅  Player features IMPROVE accuracy by {diff:.1f}pp on average.")
            print(f"  ✅  ~{avg_p:.0f} player features enter the top-60 on average.")
            print("  RECOMMENDATION: Promote player features to production pipeline.")
        elif diff >= -0.5:
            print(f"  ➡   Player features are NEUTRAL ({diff:+.1f}pp accuracy change).")
            print("  Consider log-loss and whether the added complexity is worthwhile.")
        else:
            print(f"  ❌  Player features HURT accuracy by {abs(diff):.1f}pp on average.")
            print("  Do NOT add player features to the production pipeline yet.")


def print_feature_importance(importance_df: pd.DataFrame, top_n: int = 60) -> None:
    """Print which player features ranked highest in feature selection."""
    if importance_df.empty:
        return

    print("\n" + "=" * 80)
    print("  PLAYER FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    player_imp = (
        importance_df[importance_df["is_player_feature"]]
        .groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            mean_rank=("rank", "mean"),
            times_in_top=("rank", lambda x: (x <= top_n).sum()),
            n_seasons=("year", "count"),
        )
        .sort_values("mean_importance", ascending=False)
    )

    if player_imp.empty:
        print(f"\n  No player features appeared in the top-{top_n} feature selection.")
        return

    print(f"\n  Player features by average importance across all test seasons:")
    print(f"  {'Feature':<52s}  {'Avg Imp':>8s}  {'Avg Rank':>9s}  {'In top-60':>10s}")
    print(f"  {'─' * 85}")
    for feat, row in player_imp.head(20).iterrows():
        in_top = f"{int(row['times_in_top'])}/{int(row['n_seasons'])}"
        print(f"  {feat:<52s}  {row['mean_importance']:>8.1f}  "
              f"{row['mean_rank']:>9.1f}  {in_top:>10s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest player-level form features for NRL match prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-years", nargs="+", type=int, default=[2023, 2024, 2025],
        help="Test seasons (default: 2023 2024 2025)",
    )
    parser.add_argument(
        "--min-train-year", type=int, default=2014,
        help="Earliest training year inclusive (default: 2014)",
    )
    parser.add_argument(
        "--top-n", type=int, default=60,
        help="Number of top features to select (default: 60, matching predict_round.py)",
    )
    args = parser.parse_args()

    t_total = time.time()

    print()
    print("=" * 80)
    print("  NRL Backtest — Player-Level Form Features")
    print("=" * 80)
    print(f"\n  Test seasons  : {args.test_years}")
    print(f"  Min train year: {args.min_train_year}")
    print(f"  Top-N features: {args.top_n}")

    # ── Load core data ────────────────────────────────────────────────────
    print("\n  Loading historical data...")
    matches, ladders, odds, match_stats = load_historical_data()
    elo_params = get_elo_params(matches, retune=False)

    # ── Load player stats (graceful if missing) ───────────────────────────
    player_stats_df = load_player_match_stats()

    # ── Build V4 feature matrix ───────────────────────────────────────────
    print("\n  Building V4 feature matrix (full pipeline)...")
    all_data = build_all_v4_features(matches, ladders, odds, match_stats, elo_params)
    print(f"  Feature matrix: {all_data.shape[0]} rows, {all_data.shape[1]} cols")

    # ── Add player form features if data is available ─────────────────────
    player_cols: list[str] = []
    if player_stats_df is not None:
        cols_before = set(all_data.columns)
        all_data = compute_player_form_features(all_data, player_stats_df)
        player_cols = [c for c in all_data.columns if c not in cols_before]
        for col in player_cols:
            all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
        print(f"  Player form features added: {len(player_cols)}")

    # ── Baseline feature set (V4 FEATURE_COLS filtered to existing) ───────
    baseline_cols = [c for c in FEATURE_COLS if c in all_data.columns]
    for col in baseline_cols:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")

    print(f"\n  Baseline features : {len(baseline_cols)}")
    print(f"  Player features   : {len(player_cols)}")
    print(f"  Combined total    : {len(baseline_cols) + len(player_cols)}")

    # ── Walk-forward backtest ─────────────────────────────────────────────
    out = run_backtest(
        all_data=all_data,
        baseline_cols=baseline_cols,
        player_cols=player_cols,
        test_years=args.test_years,
        min_train_year=args.min_train_year,
        top_n=args.top_n,
    )

    results_df = out["results"]
    importance_df = out["importance"]

    # ── Print comparison table ────────────────────────────────────────────
    print_comparison_table(results_df, has_player=len(player_cols) > 0)
    print_feature_importance(importance_df, top_n=args.top_n)

    # ── Save outputs ──────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_df.empty:
        out_path = out_dir / "player_features_backtest_results.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\n  Results    → {out_path}")

    if not importance_df.empty:
        imp_path = out_dir / "player_features_importance.csv"
        importance_df.to_csv(imp_path, index=False)
        print(f"  Importance → {imp_path}")

    elapsed = time.time() - t_total
    print(f"\n  Total runtime: {elapsed:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
