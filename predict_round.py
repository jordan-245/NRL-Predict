"""
NRL 2026 Match Predictor - V3 OptBlend Model
=============================================
Predicts NRL match winners using the V3 OptBlend ensemble.

The model blends 7 base models (XGBoost, LightGBM, CatBoost, LogReg,
plus top-50-feature variants) with bookmaker odds using optimized weights.

Backtested performance (walk-forward 2018-2025):
  Accuracy: 68.4%  |  Log Loss: 0.5977  |  Beats bookmaker odds on all metrics

Usage:
    python predict_round.py --auto                    # fetch from Odds API, auto-detect round
    python predict_round.py --auto --round 5          # fetch round 5 from Odds API
    python predict_round.py --auto --match "Sharks"   # single game, fresh odds (~3s)
    python predict_round.py --auto --retrain          # force retrain models
    python predict_round.py --round 1                 # manual CSV mode
    python predict_round.py --round 5 --input r5.csv  # custom CSV path
    python predict_round.py --auto --retune-elo       # API mode + retune Elo

Manual CSV format (data/upcoming/round_N.csv):
    home_team,away_team,venue,date,odds_home,odds_away
    Penrith Panthers,Sydney Roosters,BlueBet Stadium,2026-03-06,1.55,2.45

Team names must use canonical NRL names (see config/team_mappings.py).
Odds are decimal format (e.g. 1.55 = $1.55).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostClassifier
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UPCOMING_DIR = PROJECT_ROOT / "data" / "upcoming"
MODEL_CACHE_DIR = PROJECT_ROOT / "outputs" / "model_cache"
CONFIG_DIR = PROJECT_ROOT / "config"

# Import V3 feature building functions
import run_enhance_and_retrain as v3

# =====================================================================
# V3 OptBlend Model Specification
# =====================================================================

# Tuned hyperparameters (from V3 Optuna search)
BEST_XGB_PARAMS = v3.BEST_XGB_PARAMS
BEST_LGB_PARAMS = v3.BEST_LGB_PARAMS
BEST_CAT_PARAMS = v3.BEST_CAT_PARAMS

# OptBlend weights (optimized across 2018-2025 walk-forward folds)
BLEND_WEIGHTS = {
    "XGBoost":   1.025,
    "LightGBM": -0.599,
    "CatBoost":  0.281,
    "LogReg":    0.054,
    "XGB_top50": -1.126,
    "LGB_top50":  0.540,
    "CAT_top50":  0.005,
}
BLEND_ODDS_WEIGHT = 0.819  # 1 - sum(model weights)

# V3 feature column specification (141 features)
FEATURE_COLS = [
    # Elo (4)
    "home_elo", "away_elo", "home_elo_prob", "elo_diff",
    # Rolling form 3/5/8 (30)
    *[f"home_{s}_{w}" for w in [3,5,8] for s in ["win_rate","avg_pf","avg_pa","avg_margin"]],
    *[f"away_{s}_{w}" for w in [3,5,8] for s in ["win_rate","avg_pf","avg_pa","avg_margin"]],
    *[f"win_rate_diff_{w}" for w in [3,5,8]],
    *[f"avg_margin_diff_{w}" for w in [3,5,8]],
    # Ladder (10)
    "home_ladder_pos", "away_ladder_pos", "ladder_pos_diff",
    "home_wins", "away_wins", "home_losses", "away_losses",
    "home_points_diff_season", "away_points_diff_season",
    "home_competition_points", "away_competition_points",
    # Ladder home/away splits (16)
    "home_team_home_win_pct", "away_team_away_win_pct",
    "home_team_home_ppg", "away_team_away_ppg",
    "home_home_win_pct", "away_home_win_pct",
    "home_away_win_pct", "away_away_win_pct",
    "home_home_ppg", "away_home_ppg", "home_away_ppg", "away_away_ppg",
    "home_home_pag", "away_home_pag", "home_away_pag", "away_away_pag",
    # Schedule (7)
    "home_days_rest", "away_days_rest", "rest_diff",
    "home_is_back_to_back", "away_is_back_to_back",
    "home_bye_last_round", "away_bye_last_round",
    # Context (5)
    "is_home", "round_number", "is_finals", "day_of_week", "month",
    # Odds (12)
    "odds_home_prob", "odds_away_prob", "odds_home_favourite",
    "odds_home_open_prob", "odds_away_open_prob",
    "spread_home_open", "total_line_open",
    "odds_home_range", "odds_away_range",
    "bookmakers_surveyed",
    "odds_movement", "odds_movement_abs",
    # H2H (9)
    "h2h_home_win_rate_3", "h2h_home_win_rate_5", "h2h_home_win_rate_all",
    "h2h_avg_margin_3", "h2h_avg_margin_5", "h2h_avg_margin_all",
    "h2h_matches_3", "h2h_matches_5", "h2h_matches_all",
    # Venue (4)
    "home_venue_win_rate", "away_venue_win_rate",
    "venue_avg_total_score", "is_neutral_venue",
    # Momentum (10)
    "home_form_momentum", "away_form_momentum", "form_momentum_diff",
    "home_form_momentum_3v5", "away_form_momentum_3v5",
    "home_streak", "away_streak", "streak_diff",
    "home_last_result", "away_last_result",
    # Halftime / Penalty (6)
    "home_avg_halftime_lead_5", "away_avg_halftime_lead_5",
    "home_avg_penalty_diff_5", "away_avg_penalty_diff_5",
    "halftime_lead_diff", "penalty_diff_diff",
    # Engineered interactions (18)
    "elo_diff_sq", "elo_diff_abs",
    "odds_elo_diff", "odds_elo_abs_diff",
    "home_attack_defense_3", "away_attack_defense_3", "attack_defense_diff_3",
    "season_progress", "elo_diff_x_progress",
    "comp_points_ratio", "home_strength", "away_strength", "strength_diff",
    "elo_x_rest", "ladder_x_finals",
    "home_away_split_diff", "venue_wr_diff",
]


# =====================================================================
# Data Loading
# =====================================================================

def load_historical_data():
    """Load historical matches, ladders, and odds from parquet files.

    Unlike the V3 training pipeline, this keeps ALL matches regardless of
    whether they have odds data (important for 2026 in-season data).
    """
    print("  Loading historical data...")
    matches_raw = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    ladders = pd.read_parquet(PROCESSED_DIR / "ladders.parquet")
    odds = pd.read_parquet(PROCESSED_DIR / "odds.parquet")

    matches = matches_raw.copy()
    matches["date"] = pd.to_datetime(matches["parsed_date"], errors="coerce")
    matches["season"] = matches["year"].astype(int)

    # --- Fix home/away using odds (keep unmatched matches) ---
    matches["_orig_team1"] = matches["home_team"]
    matches["_orig_team2"] = matches["away_team"]
    matches["_orig_score1"] = matches["home_score"]
    matches["_orig_score2"] = matches["away_score"]
    for col in ["halftime_home", "halftime_away", "penalty_home", "penalty_away"]:
        if col in matches.columns:
            matches[f"_orig_{col}"] = matches[col]

    odds_ref = odds[["date", "home_team", "away_team"]].copy()
    odds_ref["date"] = pd.to_datetime(odds_ref["date"], errors="coerce")
    odds_set = set()
    for _, r in odds_ref.iterrows():
        d, h, a = r["date"], r["home_team"], r["away_team"]
        if pd.notna(d):
            odds_set.add((d, h, a))

    actual_home, actual_away = [], []
    actual_hs, actual_as = [], []
    match_order = []

    for _, row in matches.iterrows():
        dt, t1, t2 = row["date"], row["_orig_team1"], row["_orig_team2"]
        s1, s2 = row["_orig_score1"], row["_orig_score2"]
        found = False

        if (dt, t1, t2) in odds_set:
            actual_home.append(t1); actual_away.append(t2)
            actual_hs.append(s1); actual_as.append(s2)
            match_order.append("direct"); found = True
        elif (dt, t2, t1) in odds_set:
            actual_home.append(t2); actual_away.append(t1)
            actual_hs.append(s2); actual_as.append(s1)
            match_order.append("swapped"); found = True
        else:
            for delta in range(-2, 3):
                if delta == 0:
                    continue
                fdt = dt + pd.Timedelta(days=delta)
                if (fdt, t1, t2) in odds_set:
                    actual_home.append(t1); actual_away.append(t2)
                    actual_hs.append(s1); actual_as.append(s2)
                    match_order.append("fuzzy"); found = True; break
                elif (fdt, t2, t1) in odds_set:
                    actual_home.append(t2); actual_away.append(t1)
                    actual_hs.append(s2); actual_as.append(s1)
                    match_order.append("fuzzy_swap"); found = True; break

        if not found:
            # Keep as-is (RLP order) -- important for seasons without odds
            actual_home.append(t1); actual_away.append(t2)
            actual_hs.append(s1); actual_as.append(s2)
            match_order.append("kept")

    matches["home_team"] = actual_home
    matches["away_team"] = actual_away
    matches["home_score"] = actual_hs
    matches["away_score"] = actual_as
    matches["_match_order"] = match_order

    swap_mask = matches["_match_order"].isin(["swapped", "fuzzy_swap"])
    if "halftime_home" in matches.columns:
        matches.loc[swap_mask, "halftime_home"] = matches.loc[swap_mask, "_orig_halftime_away"]
        matches.loc[swap_mask, "halftime_away"] = matches.loc[swap_mask, "_orig_halftime_home"]
    if "penalty_home" in matches.columns:
        matches.loc[swap_mask, "penalty_home"] = matches.loc[swap_mask, "_orig_penalty_away"]
        matches.loc[swap_mask, "penalty_away"] = matches.loc[swap_mask, "_orig_penalty_home"]

    temp = [c for c in matches.columns if c.startswith("_orig_") or c == "_match_order"]
    matches = matches.drop(columns=temp, errors="ignore")

    # Sort chronologically
    def _rsort(r):
        try:
            return int(r)
        except (ValueError, TypeError):
            rs = str(r).lower()
            if "qualif" in rs: return 100
            elif "elim" in rs: return 101
            elif "semi" in rs: return 102
            elif "prelim" in rs: return 103
            elif "grand" in rs: return 104
            return 99

    matches["_rs"] = matches["round"].apply(_rsort)
    matches = matches.sort_values(["year", "_rs", "date"]).reset_index(drop=True)
    matches = matches.drop(columns=["_rs"])

    print(f"  Matches: {len(matches)}  |  Ladders: {len(ladders)}  |  Odds: {len(odds)}")
    return matches, ladders, odds


def load_upcoming_matches(csv_path: str | Path, round_num: int, year: int) -> pd.DataFrame:
    """Load upcoming matches from a user-provided CSV."""
    from config.team_mappings import standardise_team_name

    df = pd.read_csv(csv_path)

    required = ["home_team", "away_team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Standardise team names
    for col in ["home_team", "away_team"]:
        df[col] = df[col].apply(standardise_team_name)

    df["year"] = year
    df["season"] = year
    df["round"] = str(round_num)
    df["home_score"] = np.nan
    df["away_score"] = np.nan

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Map user odds columns to expected internal names
    if "odds_home" in df.columns and "odds_away" in df.columns:
        df["h2h_home"] = pd.to_numeric(df["odds_home"], errors="coerce")
        df["h2h_away"] = pd.to_numeric(df["odds_away"], errors="coerce")

    print(f"  Loaded {len(df)} upcoming matches for Round {round_num}")
    return df


# =====================================================================
# Elo Parameter Caching
# =====================================================================

def get_elo_params(matches: pd.DataFrame, retune: bool = False) -> dict:
    """Load cached Elo parameters, or tune new ones."""
    cache_path = CONFIG_DIR / "elo_params.json"

    if not retune and cache_path.exists():
        with open(cache_path) as f:
            params = json.load(f)
        print(f"  Elo params (cached): K={params['k_factor']:.1f}, "
              f"HA={params['home_advantage']:.1f}, "
              f"Reset={params['season_reset_factor']:.3f}, "
              f"MOV={params['mov_adjustment']}")
        return params

    print("  Tuning Elo parameters (50 Optuna trials)...")
    params = v3.tune_elo(matches, n_trials=50)

    with open(cache_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved Elo params to {cache_path}")

    return params


# =====================================================================
# Model Caching (for fast per-game re-scoring)
# =====================================================================

def _cache_path(round_num: int, year: int) -> Path:
    """Cache file for a given round."""
    return MODEL_CACHE_DIR / f"v3_round_{round_num}_{year}.joblib"


def _data_fingerprint() -> float:
    """Return mtime of matches.parquet for cache invalidation."""
    p = PROCESSED_DIR / "matches.parquet"
    return p.stat().st_mtime if p.exists() else 0.0


def save_model_cache(path: Path, artifacts: dict,
                     upcoming_feat: pd.DataFrame,
                     round_num: int, year: int):
    """Save trained models + feature state for fast re-scoring."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "version": 1,
        "data_mtime": _data_fingerprint(),
        "round_num": round_num,
        "year": year,
        "artifacts": artifacts,
        "upcoming_features": upcoming_feat,
    }, path, compress=3)
    print(f"  Model cache saved ({path.name})")


def load_model_cache(path: Path) -> dict | None:
    """Load cache, returning None if missing or stale."""
    if not path.exists():
        return None
    try:
        cache = joblib.load(path)
    except Exception:
        return None
    if cache.get("version") != 1:
        return None
    if cache["data_mtime"] != _data_fingerprint():
        print("  Cache stale (new results detected) — retraining...")
        return None
    return cache


# =====================================================================
# Fast Re-Scoring (cached models + fresh odds)
# =====================================================================

def _refresh_odds_in_features(cached_feat: pd.DataFrame,
                              fresh_api_df: pd.DataFrame) -> pd.DataFrame:
    """Update odds-derived features in cached features with fresh API odds."""
    feat = cached_feat.copy()
    for i, row in feat.iterrows():
        home, away = row["home_team"], row["away_team"]
        match = fresh_api_df[
            (fresh_api_df["home_team"] == home) &
            (fresh_api_df["away_team"] == away)
        ]
        if len(match) == 0:
            continue
        h2h_h = match.iloc[0].get("h2h_home")
        h2h_a = match.iloc[0].get("h2h_away")
        if pd.notna(h2h_h) and pd.notna(h2h_a) and h2h_h > 0 and h2h_a > 0:
            p_home = (1 / h2h_h) / (1 / h2h_h + 1 / h2h_a)
            feat.at[i, "odds_home_prob"] = p_home
            feat.at[i, "odds_away_prob"] = 1.0 - p_home
            feat.at[i, "odds_home_favourite"] = 1.0 if p_home > 0.5 else 0.0
            elo_prob = row.get("home_elo_prob", 0.5)
            feat.at[i, "odds_elo_diff"] = p_home - elo_prob
            feat.at[i, "odds_elo_abs_diff"] = abs(p_home - elo_prob)
    return feat


def score_with_models(artifacts: dict, upcoming_feat: pd.DataFrame) -> pd.DataFrame:
    """Score upcoming matches using pre-trained model artifacts."""
    models = artifacts["models"]
    scaler = artifacts["scaler"]
    top50 = artifacts["top50"]
    medians = artifacts["medians"]
    feature_cols = artifacts["feature_cols"]

    X_pred_raw = upcoming_feat[feature_cols].copy()

    # Fill missing using cached training medians
    bool_cols = {"home_is_back_to_back", "away_is_back_to_back",
                 "home_bye_last_round", "away_bye_last_round",
                 "is_finals", "odds_home_favourite", "is_home", "is_neutral_venue"}
    X_pred = X_pred_raw.copy()
    for col in feature_cols:
        fill_val = 0 if col in bool_cols else medians.get(col, 0)
        X_pred[col] = X_pred[col].fillna(fill_val)

    X_pred_top = X_pred[top50]
    X_pred_sc = scaler.transform(X_pred)

    predictions = {}
    for name, model in models.items():
        if name == "LogReg":
            preds = model.predict_proba(X_pred_sc)[:, 1]
        elif name.endswith("_top50"):
            preds = model.predict_proba(X_pred_top)[:, 1]
        else:
            preds = model.predict_proba(X_pred)[:, 1]
        predictions[name] = np.clip(preds, 1e-7, 1 - 1e-7)

    # Odds probability (freshly updated)
    if "odds_home_prob" in upcoming_feat.columns:
        odds_probs = upcoming_feat["odds_home_prob"].values.copy()
        odds_probs = np.where(np.isnan(odds_probs), 0.55, odds_probs)
    else:
        odds_probs = np.full(len(upcoming_feat), 0.55)
    odds_probs = np.clip(odds_probs, 1e-7, 1 - 1e-7)

    # OptBlend
    blended = np.zeros(len(upcoming_feat), dtype=float)
    for model_name, weight in BLEND_WEIGHTS.items():
        blended += weight * predictions[model_name]
    blended += BLEND_ODDS_WEIGHT * odds_probs
    blended = np.clip(blended, 0.01, 0.99)

    results = upcoming_feat[["home_team", "away_team", "venue", "date", "round"]].copy()
    results["home_win_prob"] = blended
    results["away_win_prob"] = 1.0 - blended
    results["odds_home_prob"] = odds_probs
    results["odds_away_prob"] = 1.0 - odds_probs
    results["tip"] = np.where(blended >= 0.5, results["home_team"], results["away_team"])
    results["confidence"] = np.abs(blended - 0.5) * 2
    for name, probs in predictions.items():
        results[f"model_{name}"] = probs
    return results


def _filter_match(results: pd.DataFrame, match_str: str) -> pd.DataFrame:
    """Filter predictions to matches containing the search string."""
    q = match_str.lower()
    mask = (
        results["home_team"].str.lower().str.contains(q, na=False) |
        results["away_team"].str.lower().str.contains(q, na=False)
    )
    filtered = results[mask]
    if filtered.empty:
        print(f"\n  No match found for '{match_str}'. Available:")
        for _, r in results.iterrows():
            print(f"    {r['home_team']} vs {r['away_team']}")
        return results
    return filtered


# =====================================================================
# Feature Building
# =====================================================================

def build_features(matches: pd.DataFrame, ladders: pd.DataFrame,
                   odds: pd.DataFrame, upcoming: pd.DataFrame,
                   elo_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build V3 features for all matches and return train/predict splits.

    Returns (train_features, predict_features, feature_cols).
    """
    # Link odds for historical matches
    linked = v3.link_odds(matches, odds)

    # Append upcoming matches (already have h2h_home/h2h_away from user CSV or API)
    all_matches = pd.concat([linked, upcoming], ignore_index=True)

    # Normalise dates: strip timezone info to avoid tz-naive vs tz-aware comparison
    all_matches["date"] = pd.to_datetime(all_matches["date"], utc=True, errors="coerce")
    all_matches["date"] = all_matches["date"].dt.tz_localize(None)

    all_matches = all_matches.sort_values("date").reset_index(drop=True)

    # Track which rows are upcoming (NaN scores)
    is_upcoming = all_matches["home_score"].isna()

    print(f"\n  Building V3 features for {len(all_matches)} matches "
          f"({is_upcoming.sum()} upcoming)...")

    # Run full V3 feature pipeline
    all_matches = v3.backfill_elo(all_matches, elo_params)
    all_matches = v3.compute_rolling_form_features(all_matches)
    all_matches = v3.compute_h2h_features(all_matches)
    all_matches = v3.compute_ladder_features(all_matches, ladders)
    all_matches = v3.compute_venue_features(all_matches)
    all_matches = v3.compute_odds_features(all_matches)
    all_matches = v3.compute_schedule_features(all_matches)
    all_matches = v3.compute_contextual_features(all_matches)
    all_matches = v3.compute_engineered_features(all_matches)

    # Create target
    all_matches["home_win"] = np.where(
        all_matches["home_score"] > all_matches["away_score"], 1.0,
        np.where(all_matches["home_score"] < all_matches["away_score"], 0.0, np.nan)
    )

    # Use V3 feature spec (filter to columns that exist)
    feature_cols = [c for c in FEATURE_COLS if c in all_matches.columns]

    # Ensure numeric
    for col in feature_cols:
        all_matches[col] = pd.to_numeric(all_matches[col], errors="coerce")

    # Split
    historical = all_matches[~is_upcoming].copy()
    upcoming_feat = all_matches[is_upcoming].copy()

    # Drop draws from training data
    historical = historical.dropna(subset=["home_win"]).reset_index(drop=True)

    print(f"  Training set: {len(historical)} matches  |  "
          f"Prediction set: {len(upcoming_feat)} matches")
    print(f"  Features: {len(feature_cols)} columns")

    return historical, upcoming_feat, feature_cols


# =====================================================================
# Model Training & Prediction
# =====================================================================

def fill_missing(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fill NaN using training medians; boolean flags default to 0."""
    bool_cols = {"home_is_back_to_back", "away_is_back_to_back",
                 "home_bye_last_round", "away_bye_last_round",
                 "is_finals", "odds_home_favourite", "is_home", "is_neutral_venue"}
    medians = X_train.median()
    Xtr = X_train.copy()
    Xte = X_test.copy()
    for col in X_train.columns:
        fill_val = 0 if col in bool_cols else medians.get(col, 0)
        Xtr[col] = Xtr[col].fillna(fill_val)
        Xte[col] = Xte[col].fillna(fill_val)
    return Xtr, Xte


def train_and_predict(historical: pd.DataFrame, upcoming: pd.DataFrame,
                      feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Train 7 base models on historical data, predict upcoming matches.

    Returns (results_df, artifacts_dict).  Artifacts are cached for fast
    re-scoring with updated odds.
    """
    print("\n  Training V3 OptBlend ensemble...")

    X_train_raw = historical[feature_cols].copy()
    y_train = historical["home_win"].values
    X_pred_raw = upcoming[feature_cols].copy()

    # Sample weights (exponential decay favouring recent seasons)
    train_years = historical["year"].values
    max_yr = train_years.max()
    sample_weights = 0.9 ** (max_yr - train_years)

    # Training medians (cached for fast re-scoring)
    medians = X_train_raw.median()

    # Fill missing values
    X_train, X_pred = fill_missing(X_train_raw, X_pred_raw)

    # Feature selection for top-50 variants
    print("    Selecting top-50 features...")
    selector = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.02,
                                  verbosity=0, random_state=42)
    selector.fit(X_train, y_train, sample_weight=sample_weights)
    imp = pd.Series(selector.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top50 = list(imp.head(50).index)
    X_train_top = X_train[top50]
    X_pred_top = X_pred[top50]

    predictions = {}
    trained_models = {}

    # --- XGBoost (all features) ---
    print("    Training XGBoost...")
    m = xgb.XGBClassifier(**BEST_XGB_PARAMS)
    m.fit(X_train, y_train, sample_weight=sample_weights)
    predictions["XGBoost"] = np.clip(m.predict_proba(X_pred)[:, 1], 1e-7, 1-1e-7)
    trained_models["XGBoost"] = m

    # --- XGBoost (top 50) ---
    print("    Training XGBoost (top-50)...")
    m = xgb.XGBClassifier(**BEST_XGB_PARAMS)
    m.fit(X_train_top, y_train, sample_weight=sample_weights)
    predictions["XGB_top50"] = np.clip(m.predict_proba(X_pred_top)[:, 1], 1e-7, 1-1e-7)
    trained_models["XGB_top50"] = m

    # --- LightGBM (all features) ---
    print("    Training LightGBM...")
    m = lgbm.LGBMClassifier(**BEST_LGB_PARAMS)
    m.fit(X_train, y_train, sample_weight=sample_weights)
    predictions["LightGBM"] = np.clip(m.predict_proba(X_pred)[:, 1], 1e-7, 1-1e-7)
    trained_models["LightGBM"] = m

    # --- LightGBM (top 50) ---
    print("    Training LightGBM (top-50)...")
    m = lgbm.LGBMClassifier(**BEST_LGB_PARAMS)
    m.fit(X_train_top, y_train, sample_weight=sample_weights)
    predictions["LGB_top50"] = np.clip(m.predict_proba(X_pred_top)[:, 1], 1e-7, 1-1e-7)
    trained_models["LGB_top50"] = m

    # --- CatBoost (all features) ---
    print("    Training CatBoost...")
    m = CatBoostClassifier(**BEST_CAT_PARAMS)
    m.fit(X_train, y_train, sample_weight=sample_weights)
    predictions["CatBoost"] = np.clip(m.predict_proba(X_pred)[:, 1], 1e-7, 1-1e-7)
    trained_models["CatBoost"] = m

    # --- CatBoost (top 50) ---
    print("    Training CatBoost (top-50)...")
    m = CatBoostClassifier(**BEST_CAT_PARAMS)
    m.fit(X_train_top, y_train, sample_weight=sample_weights)
    predictions["CAT_top50"] = np.clip(m.predict_proba(X_pred_top)[:, 1], 1e-7, 1-1e-7)
    trained_models["CAT_top50"] = m

    # --- Logistic Regression ---
    print("    Training LogReg...")
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_pr_sc = scaler.transform(X_pred)
    m = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    m.fit(X_tr_sc, y_train, sample_weight=sample_weights)
    predictions["LogReg"] = np.clip(m.predict_proba(X_pr_sc)[:, 1], 1e-7, 1-1e-7)
    trained_models["LogReg"] = m

    # --- Odds implied probability ---
    if "odds_home_prob" in upcoming.columns:
        odds_probs = upcoming["odds_home_prob"].values.copy()
        odds_probs = np.where(np.isnan(odds_probs), 0.55, odds_probs)
    else:
        odds_probs = np.full(len(upcoming), 0.55)
    odds_probs = np.clip(odds_probs, 1e-7, 1-1e-7)

    # --- Apply OptBlend weights ---
    print("    Blending with OptBlend weights...")
    blended = np.zeros(len(upcoming), dtype=float)
    for model_name, weight in BLEND_WEIGHTS.items():
        blended += weight * predictions[model_name]
    blended += BLEND_ODDS_WEIGHT * odds_probs
    blended = np.clip(blended, 0.01, 0.99)

    # Build results DataFrame
    results = upcoming[["home_team", "away_team", "venue", "date", "round"]].copy()
    results["home_win_prob"] = blended
    results["away_win_prob"] = 1.0 - blended
    results["odds_home_prob"] = odds_probs
    results["odds_away_prob"] = 1.0 - odds_probs
    results["tip"] = np.where(blended >= 0.5, results["home_team"], results["away_team"])
    results["confidence"] = np.abs(blended - 0.5) * 2  # 0 = coin flip, 1 = certain

    # Individual model predictions (for transparency)
    for name, probs in predictions.items():
        results[f"model_{name}"] = probs

    # Bundle model artifacts for caching
    artifacts = {
        "models": trained_models,
        "scaler": scaler,
        "top50": top50,
        "feature_cols": feature_cols,
        "medians": medians,
    }

    return results, artifacts


# =====================================================================
# Output Formatting
# =====================================================================

def format_predictions(results: pd.DataFrame, round_num: int, year: int):
    """Print predictions in a clean, tipping-competition-friendly format."""
    print()
    print("=" * 70)
    print(f"  NRL {year} - ROUND {round_num} PREDICTIONS")
    print(f"  Model: V3 OptBlend (68.4% accuracy / 0.5977 log loss)")
    print("=" * 70)

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hp = row["home_win_prob"] * 100
        ap = row["away_win_prob"] * 100
        tip = row["tip"]
        conf = row["confidence"]

        # Confidence label
        if conf >= 0.40:
            conf_label = "VERY HIGH"
        elif conf >= 0.20:
            conf_label = "HIGH"
        elif conf >= 0.10:
            conf_label = "MEDIUM"
        else:
            conf_label = "LOW"

        # Odds comparison
        odds_hp = row["odds_home_prob"] * 100
        edge = row["home_win_prob"] - row["odds_home_prob"]
        edge_str = f"Edge: {edge:+.1%}" if abs(edge) > 0.02 else ""

        print(f"\n  {home} ({hp:.1f}%)  vs  {away} ({ap:.1f}%)")

        venue = row.get("venue", "")
        date = row.get("date", "")
        if pd.notna(venue) and str(venue).strip():
            date_str = pd.to_datetime(date).strftime("%a %d %b") if pd.notna(date) else ""
            print(f"  {venue}  {date_str}")

        print(f"  >>> TIP: {tip:<30s} Confidence: {conf_label}")
        if edge_str:
            print(f"      Odds implied: {odds_hp:.1f}%  |  Model: {hp:.1f}%  |  {edge_str}")

    # Summary
    print("\n" + "-" * 70)
    print("  TIPS SUMMARY")
    print("-" * 70)
    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        tip = row["tip"]
        prob = max(row["home_win_prob"], row["away_win_prob"]) * 100
        print(f"  {home:<30s} vs {away:<30s} -> {tip} ({prob:.0f}%)")
    print("=" * 70)


def save_predictions(results: pd.DataFrame, round_num: int, year: int):
    """Save predictions to CSV."""
    output_dir = PROJECT_ROOT / "outputs" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"round_{round_num}_{year}.csv"
    results.to_csv(path, index=False)
    print(f"\n  Predictions saved to {path}")


# =====================================================================
# Main
# =====================================================================

def load_upcoming_from_api(round_num: int | None, year: int) -> tuple[pd.DataFrame, int]:
    """Fetch upcoming matches from The Odds API.

    Returns (upcoming_df, detected_round_num).
    """
    from scraping.odds_api import get_upcoming_round
    df, detected_round = get_upcoming_round(round_num=round_num, year=year)
    return df, detected_round


def main():
    parser = argparse.ArgumentParser(description="NRL Match Predictor - V3 OptBlend")
    parser.add_argument("--round", type=int, default=None,
                        help="Round number (auto-detected with --auto)")
    parser.add_argument("--year", type=int, default=2026, help="Season year (default: 2026)")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to upcoming matches CSV (default: data/upcoming/round_N.csv)")
    parser.add_argument("--auto", action="store_true",
                        help="Fetch fixtures + odds from The Odds API")
    parser.add_argument("--match", type=str, default=None,
                        help="Filter to a single match (team name substring, e.g. 'Sharks')")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retraining (ignore cache)")
    parser.add_argument("--retune-elo", action="store_true",
                        help="Re-run Elo hyperparameter optimization (slower)")
    args = parser.parse_args()

    t_start = time.time()
    year = args.year

    if args.auto:
        # --- API mode: fetch fixtures + odds automatically ---
        print()
        print("=" * 70)
        print(f"  NRL {year} Prediction Pipeline (auto mode)")
        print("=" * 70)

        # Always fetch from API first (for fresh odds)
        print("\n  STEP 1: Fetching from Odds API...")
        try:
            upcoming_api, round_num = load_upcoming_from_api(args.round, year)
        except ValueError as e:
            print(f"\n  ERROR: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n  ERROR: API request failed: {e}")
            print("  Check your ODDS_API_KEY in .env and try again.")
            print("  Or use manual CSV mode: python predict_round.py --round N")
            sys.exit(1)

        print(f"\n  Round {round_num}: {len(upcoming_api)} matches")

        # Try fast path (cached models + fresh odds)
        cp = _cache_path(round_num, year)
        cache = None if (args.retrain or args.retune_elo) else load_model_cache(cp)

        if cache is not None:
            # === FAST PATH: refresh odds, re-score with cached models ===
            print("\n  STEP 2: Re-scoring with fresh odds (cached models)")
            upcoming_feat = _refresh_odds_in_features(
                cache["upcoming_features"], upcoming_api
            )
            results = score_with_models(cache["artifacts"], upcoming_feat)

            if args.match:
                results = _filter_match(results, args.match)

            format_predictions(results, round_num, year)
            save_predictions(results, round_num, year)

            elapsed = time.time() - t_start
            print(f"\n  Completed in {elapsed:.0f}s (fast mode)")
            return

        # === FULL PATH: build features, train, predict, cache ===
        print("\n  STEP 2: Loading historical data")
        matches, ladders, odds = load_historical_data()

        print("\n  STEP 3: Elo parameters")
        elo_params = get_elo_params(matches, retune=args.retune_elo)

        print("\n  STEP 4: Feature engineering")
        historical, upcoming_feat, feature_cols = build_features(
            matches, ladders, odds, upcoming_api, elo_params
        )

        print("\n  STEP 5: Model training & prediction")
        results, artifacts = train_and_predict(historical, upcoming_feat, feature_cols)

        # Save cache for fast re-scoring
        save_model_cache(cp, artifacts, upcoming_feat, round_num, year)

        if args.match:
            results = _filter_match(results, args.match)

        format_predictions(results, round_num, year)
        save_predictions(results, round_num, year)

    else:
        # --- Manual CSV mode (existing behavior) ---
        if args.round is None:
            print("\n  ERROR: --round is required when not using --auto")
            print("  Usage: python predict_round.py --round N")
            print("     or: python predict_round.py --auto")
            sys.exit(1)

        round_num = args.round

        if args.input:
            csv_path = Path(args.input)
        else:
            csv_path = UPCOMING_DIR / f"round_{round_num}.csv"

        if not csv_path.exists():
            print(f"\n  ERROR: Input file not found: {csv_path}")
            print(f"\n  Create a CSV with columns: home_team,away_team,venue,date,odds_home,odds_away")
            print(f"  Save it to: {UPCOMING_DIR / f'round_{round_num}.csv'}")
            print(f"\n  Example:")
            print(f"    home_team,away_team,venue,date,odds_home,odds_away")
            print(f"    Penrith Panthers,Sydney Roosters,BlueBet Stadium,2026-03-06,1.55,2.45")
            sys.exit(1)

        print()
        print("=" * 70)
        print(f"  NRL {year} Round {round_num} Prediction Pipeline")
        print("=" * 70)

        # Step 1: Load data
        print("\n  STEP 1: Loading data")
        matches, ladders, odds = load_historical_data()
        upcoming = load_upcoming_matches(csv_path, round_num, year)

        # Step 2: Get Elo parameters
        print("\n  STEP 2: Elo parameters")
        elo_params = get_elo_params(matches, retune=args.retune_elo)

        # Step 3: Build features
        print("\n  STEP 3: Feature engineering")
        historical, upcoming_feat, feature_cols = build_features(
            matches, ladders, odds, upcoming, elo_params
        )

        # Step 4: Train and predict
        print("\n  STEP 4: Model training & prediction")
        results, _artifacts = train_and_predict(historical, upcoming_feat, feature_cols)

        if args.match:
            results = _filter_match(results, args.match)

        format_predictions(results, round_num, year)
        save_predictions(results, round_num, year)

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
