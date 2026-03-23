"""
NRL 2026 Tipping Comp Predictor
================================
Simple tipping strategy: always tip the odds favourite, use model to
break ties on close games, margin from the spread. 10 min per round.

Strategy:
  1. Clear favourites (>7% prob gap): tip odds favourite
  2. Close games (<7% prob gap): use V4 OptBlend model as tiebreaker
  3. Margin: from bookmaker spread (e.g. spread=-5.5 → margin=6)
  4. Game day: flip tip if starting halfback/fullback scratched from close game

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
from sklearn.isotonic import IsotonicRegression
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

# Import feature building functions
from pipelines import v3
from pipelines import v4

# =====================================================================
# Model Specification: V4 GBM params + V5-tuned RF/weights
# =====================================================================

BEST_XGB_PARAMS = v4.BEST_XGB_PARAMS
BEST_LGB_PARAMS = v4.BEST_LGB_PARAMS
BEST_CAT_PARAMS = v4.BEST_CAT_PARAMS
# V5-tuned RF params (improved from 0.6066 to 0.6053 log loss)
BEST_RF_PARAMS = {
    'n_estimators': 234, 'max_depth': 10, 'min_samples_leaf': 23,
    'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1,
}
SAMPLE_WEIGHT_DECAY = 0.85  # Backtested: 0.85 → 69.1% vs 0.92 → 68.1% on 1503 games (walk-forward)

# V4 OptBlend weights (optimized across 2018-2025 walk-forward folds)
BLEND_WEIGHTS = v4.V4_BLEND_WEIGHTS
BLEND_ODDS_WEIGHT = v4.V4_BLEND_ODDS_WEIGHT

# V4 feature column specification (194 features)
FEATURE_COLS = [
    # === V3 BASE FEATURES (131) ===
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
    # V3 Engineered interactions (18)
    "elo_diff_sq", "elo_diff_abs",
    "odds_elo_diff", "odds_elo_abs_diff",
    "home_attack_defense_3", "away_attack_defense_3", "attack_defense_diff_3",
    "season_progress", "elo_diff_x_progress",
    "comp_points_ratio", "home_strength", "away_strength", "strength_diff",
    "elo_x_rest", "ladder_x_finals",
    "home_away_split_diff", "venue_wr_diff",
    # === V4 NEW FEATURES (~63) ===
    # V4 Enhanced Odds (13)
    "spread_home_close", "spread_movement", "spread_movement_abs",
    "odds_spread_agree", "odds_spread_disagree_mag",
    "total_line_close", "total_movement", "total_movement_abs",
    "implied_draw_prob", "draw_competitiveness",
    "odds_home_range_close", "odds_away_range_close",
    "market_confidence",
    # V4 Scoring Consistency & Trends (31)
    *[f"{side}_{stat}_{w}" for w in [5,8] for side in ["home","away"]
      for stat in ["pf_std","pa_std","close_game_rate","pf_trend","pa_trend"]],
    *[f"{stat}_{w}" for w in [5,8]
      for stat in ["pf_std_diff","close_game_diff","pf_trend_diff","pa_trend_diff"]],
    "home_scoring_cv_5", "away_scoring_cv_5", "scoring_cv_diff_5",
    # V4 Attendance (2)
    "attendance_normalized", "attendance_high",
    # V4 Kickoff (3)
    "is_night_game", "is_afternoon_game", "is_day_game",
    # V4 Lineup Stability (3)
    "home_lineup_stability", "away_lineup_stability", "lineup_stability_diff",
    # V4 Player Impact (6)
    # Player impact features removed from training — look-ahead bias.
    # Impact is applied via --check-lineups post-prediction adjustment.
    # "home_spine_impact", "away_spine_impact", "spine_impact_diff",
    # "home_total_impact", "away_total_impact", "total_impact_diff",
    # V4 Engineered Interactions (14)
    "is_early_season", "is_mid_season", "is_late_season",
    "elo_diff_x_late", "elo_diff_x_early", "form_x_late",
    "home_defense_improving", "away_defense_improving", "defense_trend_diff",
    "scoring_env_ratio", "fav_consistency",
    "elo_spread_agree", "strong_team_rested", "home_ground_x_form",
    # V4 Team Season Stats — prior-season averages (78)
    *[f"home_ts_{s}" for s in [
        "line_breaks_average", "tackle_breaks_average", "possession_pct_average",
        "set_completion_pct_average", "all_run_metres_average",
        "post_contact_metres_average", "offloads_average", "errors_average",
        "penalties_conceded_average", "missed_tackles_average",
        "ineffective_tackles_average", "handling_errors_average",
        "intercepts_average", "kick_return_metres_average", "points_average",
        "tries_average", "try_assists_average", "tackles_average",
        "conversion_pct_average", "line_engaged_average", "supports_average",
        "all_runs_average", "all_receipts_average", "goals_average",
        "decoy_runs_average", "dummy_half_runs_average",
    ]],
    *[f"away_ts_{s}" for s in [
        "line_breaks_average", "tackle_breaks_average", "possession_pct_average",
        "set_completion_pct_average", "all_run_metres_average",
        "post_contact_metres_average", "offloads_average", "errors_average",
        "penalties_conceded_average", "missed_tackles_average",
        "ineffective_tackles_average", "handling_errors_average",
        "intercepts_average", "kick_return_metres_average", "points_average",
        "tries_average", "try_assists_average", "tackles_average",
        "conversion_pct_average", "line_engaged_average", "supports_average",
        "all_runs_average", "all_receipts_average", "goals_average",
        "decoy_runs_average", "dummy_half_runs_average",
    ]],
    *[f"ts_diff_{s}" for s in [
        "line_breaks_average", "tackle_breaks_average", "possession_pct_average",
        "set_completion_pct_average", "all_run_metres_average",
        "post_contact_metres_average", "offloads_average", "errors_average",
        "penalties_conceded_average", "missed_tackles_average",
        "ineffective_tackles_average", "handling_errors_average",
        "intercepts_average", "kick_return_metres_average", "points_average",
        "tries_average", "try_assists_average", "tackles_average",
        "conversion_pct_average", "line_engaged_average", "supports_average",
        "all_runs_average", "all_receipts_average", "goals_average",
        "decoy_runs_average", "dummy_half_runs_average",
    ]],
    # V4 Referee features (3)
    "ref_home_win_rate", "ref_games", "ref_is_high_home",
    # V4 Rolling match stats features (60)
    # 10 stats × 2 windows × 3 (home/away/diff) = 60 features
    *[f"home_ms_{s}_{w}" for w in [3, 5] for s in [
        "completion_rate", "line_breaks", "tackle_breaks", "errors", "missed_tackles",
        "all_run_metres", "possession_pct", "effective_tackle_pct", "post_contact_metres", "offloads",
    ]],
    *[f"away_ms_{s}_{w}" for w in [3, 5] for s in [
        "completion_rate", "line_breaks", "tackle_breaks", "errors", "missed_tackles",
        "all_run_metres", "possession_pct", "effective_tackle_pct", "post_contact_metres", "offloads",
    ]],
    *[f"ms_diff_{s}_{w}" for w in [3, 5] for s in [
        "completion_rate", "line_breaks", "tackle_breaks", "errors", "missed_tackles",
        "all_run_metres", "possession_pct", "effective_tackle_pct", "post_contact_metres", "offloads",
    ]],
    # V4 Player form features (42)
    # Spine form: rolling 3/5-game avg of 4 spine players' key stats (24)
    *[f"{side}_spine_{stat}_{w}" for w in [3, 5]
      for side in ["home", "away"]
      for stat in ["run_metres", "line_breaks", "tackle_breaks", "try_assists"]],
    *[f"spine_diff_{stat}_{w}" for w in [3, 5]
      for stat in ["run_metres", "line_breaks", "tackle_breaks", "try_assists"]],
    # Squad quality: rolling 3/5-game avg of starting 13's stats (12)
    *[f"{side}_squad_{stat}_{w}" for w in [3, 5]
      for side in ["home", "away"]
      for stat in ["fantasy", "minutes"]],
    *[f"squad_diff_{stat}_{w}" for w in [3, 5]
      for stat in ["fantasy", "minutes"]],
    # Squad disruption: player changes vs previous game (6)
    "home_spine_changes", "away_spine_changes", "spine_changes_diff",
    "home_squad_turnover", "away_squad_turnover", "squad_turnover_diff",
    # === V4.1 FEATURES (ablation-validated) ===
    # Travel distance (5) — ablation: +1 tip
    "home_travel_km", "away_travel_km", "travel_diff_km",
    "away_is_interstate", "away_is_overseas",
    # Opponent-adjusted rolling stats (15) — ablation: +2 tips
    *[f"home_oa_{s}_5" for s in [
        "completion_rate", "line_breaks", "errors", "all_run_metres", "missed_tackles",
    ]],
    *[f"away_oa_{s}_5" for s in [
        "completion_rate", "line_breaks", "errors", "all_run_metres", "missed_tackles",
    ]],
    *[f"oa_diff_{s}_5" for s in [
        "completion_rate", "line_breaks", "errors", "all_run_metres", "missed_tackles",
    ]],
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
    odds_path = PROCESSED_DIR / "odds.parquet"
    if odds_path.exists():
        odds = pd.read_parquet(odds_path)
    else:
        print("  WARNING: odds.parquet not found — running without historical odds")
        odds = pd.DataFrame(columns=["date", "home_team", "away_team"])

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

    match_stats_path = PROCESSED_DIR / "match_stats.parquet"
    if match_stats_path.exists():
        match_stats = pd.read_parquet(match_stats_path)
    else:
        print("  WARNING: match_stats.parquet not found — running without rolling match stats")
        match_stats = None

    player_stats_path = PROCESSED_DIR / "player_match_stats.parquet"
    if player_stats_path.exists():
        player_match_stats = pd.read_parquet(player_stats_path)
    else:
        print("  WARNING: player_match_stats.parquet not found — running without player form features")
        player_match_stats = None

    print(f"  Matches: {len(matches)}  |  Ladders: {len(ladders)}  |  Odds: {len(odds)}"
          f"  |  Match stats: {len(match_stats) if match_stats is not None else 0}"
          f"  |  Player stats: {len(player_match_stats) if player_match_stats is not None else 0}")
    return matches, ladders, odds, match_stats, player_match_stats


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

    # Tag as user-supplied upcoming match (vs scraped future fixtures)
    df["_is_user_upcoming"] = True

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
            feat.at[i, "h2h_home"] = h2h_h
            feat.at[i, "h2h_away"] = h2h_a
        # Update spread from fresh API data
        spread = match.iloc[0].get("spread_home")
        if pd.notna(spread):
            feat.at[i, "spread_home"] = spread
    return feat


def _get_round_blend_weights(round_number: int) -> tuple[float, float]:
    """Get model/odds blend weights.

    Fixed 35/65 for all rounds.  Dynamic R1-3 dampening was tested but
    with decay=0.85 the model already handles early rounds well (66% vs
    64% with old decay).  Adding a 15% dampening on top actually hurt
    by -4 tips.  The lower decay naturally solves the early-season problem
    by down-weighting stale historical data.

    Returns (model_weight, odds_weight).
    """
    return 0.35, 0.65


def score_with_models(artifacts: dict, upcoming_feat: pd.DataFrame) -> pd.DataFrame:
    """Score upcoming matches using pre-trained model artifacts."""
    models = artifacts["models"]
    scaler = artifacts["scaler"]
    top50 = artifacts["top50"]
    medians = artifacts["medians"]
    feature_cols = artifacts["feature_cols"]

    X_pred_raw = upcoming_feat[feature_cols].copy()

    # Sanitize Round 1 outliers + fill odds features coherently
    X_pred = _sanitize_round1_features(X_pred_raw.copy())
    X_pred = _fill_odds_coherent(X_pred)

    # Then fill remaining NaNs with cached training medians
    bool_cols = {"home_is_back_to_back", "away_is_back_to_back",
                 "home_bye_last_round", "away_bye_last_round",
                 "is_finals", "odds_home_favourite", "is_home", "is_neutral_venue"}
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

    # --- Dynamic round-based blend weights ---
    # Early rounds have limited current-season signal; lean more on market odds.
    if "round" in upcoming_feat.columns and len(upcoming_feat) > 0:
        try:
            round_number = int(pd.to_numeric(upcoming_feat["round"], errors="coerce").median())
        except (ValueError, TypeError):
            round_number = 6
    else:
        round_number = 6
    model_weight, odds_weight = _get_round_blend_weights(round_number)
    print(f"    Blend weights (round {round_number}): "
          f"model={model_weight:.0%}, odds={odds_weight:.0%}")

    cat_pred = predictions.get("CAT_top50", np.full(len(upcoming_feat), 0.5))

    # Meta-learner blend (if cached, for rounds 6+) or dynamic linear blend
    meta_lr = artifacts.get("meta_lr")
    if meta_lr is not None and round_number >= 6:
        # Regular season: use meta-learner (learned optimal non-linear blend)
        X_meta_pred = pd.DataFrame({
            "model_prob": cat_pred,
            "odds_prob": odds_probs,
            "model_x_odds": cat_pred * odds_probs,
            "prob_diff": cat_pred - odds_probs,
            "abs_diff": np.abs(cat_pred - odds_probs),
            "confidence": np.maximum(odds_probs, 1 - odds_probs),
        }).fillna(0.5)
        blended = np.clip(meta_lr.predict_proba(X_meta_pred)[:, 1], 0.01, 0.99)
    else:
        # Early rounds (1-5) or no meta-learner: explicit dynamic linear blend.
        # Dampens model signal when current-season data is thin.
        blended = model_weight * cat_pred + odds_weight * odds_probs
        blended = np.clip(blended, 0.01, 0.99)

    # --- Isotonic calibration (fixes overconfidence in 0.7-0.8 range) ---
    calibrator = artifacts.get("calibrator")
    if calibrator is not None:
        blended = calibrator.predict(blended)
        blended = np.clip(blended, 0.01, 0.99)

    results = upcoming_feat[["home_team", "away_team", "venue", "date", "round"]].copy()
    results["home_win_prob"] = blended
    results["away_win_prob"] = 1.0 - blended
    results["odds_home_prob"] = odds_probs
    results["odds_away_prob"] = 1.0 - odds_probs
    results["tip"] = np.where(blended >= 0.5, results["home_team"], results["away_team"])
    results["confidence"] = np.abs(blended - 0.5) * 2
    # Carry through spread and decimal odds for tipping sheet
    for col in ["spread_home", "h2h_home", "h2h_away"]:
        if col in upcoming_feat.columns:
            results[col] = upcoming_feat[col].values
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
                   elo_params: dict,
                   match_stats: pd.DataFrame | None = None,
                   player_match_stats: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build V4 features for all matches and return train/predict splits.

    Returns (train_features, predict_features, feature_cols).
    """
    # Link odds for historical matches
    linked = v3.link_odds(matches, odds)

    # Remove unplayed matches from historical data (e.g. future 2026 fixtures
    # scraped from RLP that have NaN scores — they add no training value and
    # confuse feature computation / upcoming match identification)
    before = len(linked)
    linked = linked.dropna(subset=["home_score"]).reset_index(drop=True)
    dropped = before - len(linked)
    if dropped > 0:
        print(f"  Dropped {dropped} unplayed historical matches (NaN scores)")

    # Append upcoming matches (already have h2h_home/h2h_away from user CSV or API)
    all_matches = pd.concat([linked, upcoming], ignore_index=True)

    # Normalise dates: strip timezone info to avoid tz-naive vs tz-aware comparison
    all_matches["date"] = pd.to_datetime(all_matches["date"], utc=True, errors="coerce")
    all_matches["date"] = all_matches["date"].dt.tz_localize(None)

    all_matches = all_matches.sort_values("date").reset_index(drop=True)

    # Tag user-supplied upcoming matches (not just any NaN-score match)
    # The upcoming df has a special marker; 2026 fixtures from RLP also have
    # NaN scores but should NOT be predicted.
    if "_is_user_upcoming" not in all_matches.columns:
        all_matches["_is_user_upcoming"] = False
    is_upcoming = all_matches["_is_user_upcoming"].fillna(False).astype(bool)

    print(f"\n  Building features for {len(all_matches)} matches "
          f"({is_upcoming.sum()} upcoming)...")

    # Run full V3 base feature pipeline
    all_matches = v3.backfill_elo(all_matches, elo_params)
    all_matches = v3.compute_rolling_form_features(all_matches)
    all_matches = v3.compute_h2h_features(all_matches)
    all_matches = v3.compute_ladder_features(all_matches, ladders)
    all_matches = v3.compute_venue_features(all_matches)
    all_matches = v3.compute_odds_features(all_matches)
    all_matches = v3.compute_schedule_features(all_matches)
    all_matches = v3.compute_contextual_features(all_matches)
    all_matches = v3.compute_engineered_features(all_matches)

    # Run V4 enhanced feature pipeline
    all_matches = v4.compute_v4_odds_features(all_matches)
    all_matches = v4.compute_scoring_consistency_features(all_matches)
    all_matches = v4.compute_attendance_features(all_matches)
    all_matches = v4.compute_kickoff_features(all_matches)
    all_matches = v4.compute_lineup_stability_features(all_matches)
    all_matches = v4.compute_player_impact_features(all_matches)
    all_matches = v4.compute_team_stats_features(all_matches)
    all_matches = v4.compute_referee_features(all_matches)
    all_matches = v4.compute_v4_engineered_features(all_matches)

    # Rolling per-game match stats (process quality: completion rate, line breaks, etc.)
    all_matches = v4.compute_rolling_match_stats_features(all_matches, match_stats)

    # Player-level form features (spine form, squad quality, disruption)
    all_matches = v4.compute_player_form_features(all_matches, player_match_stats)

    # V4.1 features (ablation-validated: travel +1 tip, opponent-adjusted +2 tips)
    try:
        from features.travel import compute_travel_features
        all_matches = compute_travel_features(all_matches)
    except ImportError:
        pass

    try:
        from features.opponent_adjusted import compute_opponent_adjusted_features
        all_matches = compute_opponent_adjusted_features(all_matches, match_stats)
    except ImportError:
        pass

    # Create target
    all_matches["home_win"] = np.where(
        all_matches["home_score"] > all_matches["away_score"], 1.0,
        np.where(all_matches["home_score"] < all_matches["away_score"], 0.0, np.nan)
    )

    # Use V4 feature spec (filter to columns that exist)
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

def _sanitize_round1_features(X: pd.DataFrame) -> pd.DataFrame:
    """Clamp out-of-distribution features that appear in season openers.

    Round 1 games have ~175 days rest (off-season), all teams flagged as
    'bye last round', and extreme carry-over form stats.  These values are
    far outside the training distribution, causing tree models to
    extrapolate unpredictably.
    """
    X = X.copy()

    # Cap days rest at 14 (treat off-season as a regular bye, not 175 days)
    MAX_REST = 14
    for col in ["home_days_rest", "away_days_rest"]:
        if col in X.columns:
            X[col] = X[col].clip(upper=MAX_REST)
    if "rest_diff" in X.columns and "home_days_rest" in X.columns and "away_days_rest" in X.columns:
        X["rest_diff"] = X["home_days_rest"] - X["away_days_rest"]

    # Off-season is not a bye — clear these flags for Round 1
    if "round_number" in X.columns:
        is_r1 = X["round_number"] == 1
        for col in ["home_bye_last_round", "away_bye_last_round"]:
            if col in X.columns:
                X.loc[is_r1, col] = 0

        # Lineup stability is meaningless at Round 1 — off-season roster
        # changes ≠ injury disruption.  Set to NaN → median fill (neutral).
        for col in ["home_lineup_stability", "away_lineup_stability",
                     "lineup_stability_diff"]:
            if col in X.columns:
                X.loc[is_r1, col] = np.nan

    return X


def _fill_odds_coherent(X: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN odds features coherently from actual closing odds.

    When the Odds API only returns h2h closing prices, derived features
    (opening odds, spreads, movement) are NaN.  Filling those with
    training medians creates phantom signals (e.g. median says 'home
    favourite' when the real odds say 'away favourite').  Instead, we
    derive them from the available closing odds so all odds features
    tell a consistent story.
    """
    X = X.copy()
    hp = X.get("odds_home_prob")
    ap = X.get("odds_away_prob")
    if hp is None or ap is None:
        return X

    # Opening odds → mirror closing (assume no movement)
    for col, src in [("odds_home_open_prob", hp),
                     ("odds_away_open_prob", ap)]:
        if col in X.columns:
            X[col] = X[col].fillna(src)

    # Spread → derive from odds prob  (approx: margin ≈ (prob-0.5) * 26)
    if "spread_home_open" in X.columns:
        derived_spread = -((hp - 0.5) * 26)
        X["spread_home_open"] = X["spread_home_open"].fillna(derived_spread)
    if "spread_home_close" in X.columns:
        derived_spread = -((hp - 0.5) * 26)
        X["spread_home_close"] = X["spread_home_close"].fillna(derived_spread)

    # Total line → use training median (neutral, no directional bias)
    # (left for generic median fill below)

    # Movement features → 0 (no movement when open == close)
    for col in ["odds_movement", "odds_movement_abs",
                "spread_movement", "spread_movement_abs",
                "total_movement", "total_movement_abs"]:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    # Range/bookmaker features → neutral values
    for col in ["odds_home_range", "odds_away_range",
                "odds_home_range_close", "odds_away_range_close"]:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    if "bookmakers_surveyed" in X.columns:
        X["bookmakers_surveyed"] = X["bookmakers_surveyed"].fillna(1)

    # Draw/competitiveness → neutral
    for col in ["implied_draw_prob", "draw_competitiveness"]:
        if col in X.columns:
            X[col] = X[col].fillna(0.05)

    return X


def fill_missing(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fill NaN using training medians; odds features filled coherently."""
    bool_cols = {"home_is_back_to_back", "away_is_back_to_back",
                 "home_bye_last_round", "away_bye_last_round",
                 "is_finals", "odds_home_favourite", "is_home", "is_neutral_venue"}
    medians = X_train.median()

    # Sanitize Round 1 outliers + fill odds features coherently
    Xtr = _sanitize_round1_features(X_train.copy())
    Xtr = _fill_odds_coherent(Xtr)
    Xte = _sanitize_round1_features(X_test.copy())
    Xte = _fill_odds_coherent(Xte)

    for col in X_train.columns:
        med = medians.get(col, 0)
        if pd.isna(med):
            med = 0
        fill_val = 0 if col in bool_cols else med
        Xtr[col] = Xtr[col].fillna(fill_val)
        Xte[col] = Xte[col].fillna(fill_val)
    return Xtr, Xte


def train_and_predict(historical: pd.DataFrame, upcoming: pd.DataFrame,
                      feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Train 7 base models on historical data, predict upcoming matches.

    Returns (results_df, artifacts_dict).  Artifacts are cached for fast
    re-scoring with updated odds.
    """
    print("\n  Training CAT_top50 + Odds blend...")

    X_train_raw = historical[feature_cols].copy()
    y_train = historical["home_win"].values
    X_pred_raw = upcoming[feature_cols].copy()

    # Sample weights (exponential decay favouring recent seasons)
    train_years = historical["year"].values
    max_yr = train_years.max()
    sample_weights = np.asarray(SAMPLE_WEIGHT_DECAY ** (max_yr - train_years), dtype=np.float64)

    # Training medians (cached for fast re-scoring)
    medians = X_train_raw.median()

    # Fill missing values
    X_train, X_pred = fill_missing(X_train_raw, X_pred_raw)

    # Feature selection — top-60 by XGBoost importance
    # (expanded from top-50 to capture team season stats features)
    TOP_N_FEATURES = 60
    print(f"    Selecting top-{TOP_N_FEATURES} features...")
    selector = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.02,
                                  verbosity=0, random_state=42)
    selector.fit(X_train, y_train, sample_weight=sample_weights)
    imp = pd.Series(selector.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top50 = list(imp.head(TOP_N_FEATURES).index)
    X_train_top = X_train[top50]
    X_pred_top = X_pred[top50]

    predictions = {}
    trained_models = {}

    # --- CatBoost (top features) — sole model in blend ---
    print(f"    Training CatBoost on {len(top50)} features...")
    m = CatBoostClassifier(**BEST_CAT_PARAMS)
    m.fit(X_train_top, y_train, sample_weight=sample_weights)
    predictions["CAT_top50"] = np.clip(m.predict_proba(X_pred_top)[:, 1], 1e-7, 1-1e-7)
    trained_models["CAT_top50"] = m

    # Scaler (kept for cache compatibility)
    scaler = StandardScaler()
    scaler.fit(X_train)

    # --- Odds implied probability ---
    if "odds_home_prob" in upcoming.columns:
        odds_probs = upcoming["odds_home_prob"].values.copy()
        odds_probs = np.where(np.isnan(odds_probs), 0.55, odds_probs)
    else:
        odds_probs = np.full(len(upcoming), 0.55)
    odds_probs = np.clip(odds_probs, 1e-7, 1-1e-7)

    # --- Meta-learner: LR that learns optimal non-linear blend ---
    # Train on cross-validated model predictions from training data
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    print("    Training meta-learner (LR stacking)...")
    # Get out-of-fold model predictions on training data
    # Compute out-of-fold predictions via manual CV (CatBoost needs sample_weight)
    from sklearn.model_selection import KFold
    cat_oof = np.zeros(len(y_train))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold_train, fold_val in kf.split(X_train_top):
        fold_model = CatBoostClassifier(**{**BEST_CAT_PARAMS, "verbose": 0})
        fold_model.fit(
            X_train_top.iloc[fold_train], y_train[fold_train],
            sample_weight=sample_weights[fold_train],
        )
        cat_oof[fold_val] = fold_model.predict_proba(X_train_top.iloc[fold_val])[:, 1]

    train_odds = historical["odds_home_prob"].values.copy()
    train_odds = np.where(np.isnan(train_odds), 0.55, train_odds)

    # Build meta-features
    X_meta_train = pd.DataFrame({
        "model_prob": cat_oof,
        "odds_prob": train_odds,
        "model_x_odds": cat_oof * train_odds,
        "prob_diff": cat_oof - train_odds,
        "abs_diff": np.abs(cat_oof - train_odds),
        "confidence": np.maximum(train_odds, 1 - train_odds),
    }).fillna(0.5)

    meta_lr = LogisticRegression(C=1.0, max_iter=1000)
    meta_lr.fit(X_meta_train, y_train)

    # --- Isotonic calibration: fit on OOF blended predictions ---
    # Blends CatBoost OOF + train odds at default weights to get training-time blend.
    # IsotonicRegression learns a monotone correction that fixes overconfidence
    # (e.g. predicted 0.75 only wins 67% of the time → calibrated to 0.67).
    print("    Fitting isotonic calibrator on OOF predictions...")
    blended_oof = 0.35 * cat_oof + 0.65 * train_odds
    blended_oof = np.clip(blended_oof, 0.01, 0.99)
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(blended_oof, y_train)
    print(f"    Calibrator fitted on {len(y_train)} training samples")

    # Apply meta-learner to prediction set
    cat_pred = predictions["CAT_top50"]
    X_meta_pred = pd.DataFrame({
        "model_prob": cat_pred,
        "odds_prob": odds_probs,
        "model_x_odds": cat_pred * odds_probs,
        "prob_diff": cat_pred - odds_probs,
        "abs_diff": np.abs(cat_pred - odds_probs),
        "confidence": np.maximum(odds_probs, 1 - odds_probs),
    }).fillna(0.5)

    meta_blended = meta_lr.predict_proba(X_meta_pred)[:, 1]

    # Fallback: also compute simple linear blend
    blended_linear = np.zeros(len(upcoming), dtype=float)
    for model_name, weight in BLEND_WEIGHTS.items():
        blended_linear += weight * predictions[model_name]
    blended_linear += BLEND_ODDS_WEIGHT * odds_probs

    # Use meta-learner as primary, with safety clip
    blended = np.clip(meta_blended, 0.01, 0.99)

    # Build results DataFrame
    results = upcoming[["home_team", "away_team", "venue", "date", "round"]].copy()
    results["home_win_prob"] = blended
    results["away_win_prob"] = 1.0 - blended
    results["odds_home_prob"] = odds_probs
    results["odds_away_prob"] = 1.0 - odds_probs
    results["tip"] = np.where(blended >= 0.5, results["home_team"], results["away_team"])
    results["confidence"] = np.abs(blended - 0.5) * 2  # 0 = coin flip, 1 = certain
    # Carry through spread and decimal odds for tipping sheet
    for col in ["spread_home", "h2h_home", "h2h_away"]:
        if col in upcoming.columns:
            results[col] = upcoming[col].values

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
        "meta_lr": meta_lr,
        "calibrator": calibrator,
    }

    return results, artifacts


# =====================================================================
# Lineup Check & Player Impact Adjustment
# =====================================================================

def check_lineups_and_adjust(
    results: pd.DataFrame,
    round_num: int,
    year: int,
) -> pd.DataFrame:
    """Fetch NRL.com team lists, diff vs expected starters, adjust probabilities.

    For each team, identifies missing expected starters and adjusts the
    home_win_prob using player impact scores.
    """
    from scraping.nrl_teamlists import (
        fetch_round_teamlists, diff_lineups, get_expected_starters,
        load_baseline, diff_against_baseline, save_baseline,
    )
    from processing.player_impact import get_player_impact, OUTPUT_PATH as IMPACT_PATH

    print("\n  LINEUP CHECK: Fetching NRL.com team lists...")

    # Load player impact data
    impact_df = None
    if IMPACT_PATH.exists():
        impact_df = pd.read_parquet(IMPACT_PATH)
        print(f"    Loaded {len(impact_df)} player impact scores")
    else:
        print("    WARNING: No player impact data — run: python -m processing.player_impact")
        print("    Lineup check will show changes but cannot adjust probabilities")

    # Fetch current team lists from NRL.com
    teamlists = fetch_round_teamlists(year, round_num, use_cache=False, delay=0.5)

    if not teamlists:
        print("    No team lists available from NRL.com")
        return results

    # Build lookup: team → current players
    current_by_team = {}
    for tl in teamlists:
        current_by_team[tl["home_team"]] = tl["home_players"]
        current_by_team[tl["away_team"]] = tl["away_players"]

    print(f"    Got team lists for {len(teamlists)} matches")

    # Load baseline teamlists (captured during Tuesday tips pipeline).
    # If a baseline exists, compare against it (correct: detects game-day
    # scratches only).  If not, create one now and fall back to historical
    # appearances (legacy mode).
    baseline = load_baseline(year, round_num)
    use_baseline = baseline is not None
    if use_baseline:
        print(f"    Using baseline teamlists ({len(baseline)} teams)")
    else:
        print("    No baseline found — saving current as baseline, using legacy diff")
        save_baseline(year, round_num, teamlists)

    # Legacy fallback: load historical appearances
    appearances_df = None
    if not use_baseline:
        app_path = PROCESSED_DIR / "player_appearances.parquet"
        if app_path.exists():
            appearances_df = pd.read_parquet(app_path)

    print()

    # Process each match
    results = results.copy()
    lineup_alerts = []

    for idx, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_adj = 0.0
        away_adj = 0.0
        match_alerts = []

        for team, side in [(home, "home"), (away, "away")]:
            current_players = current_by_team.get(team, [])
            if not current_players:
                continue

            if use_baseline:
                # Compare against Tuesday-announced squad (correct baseline)
                changes = diff_against_baseline(team, current_players, baseline)
            else:
                # Fallback: compare against historical appearances
                expected = get_expected_starters(team, appearances_df, n_recent=5)
                if not expected:
                    continue
                changes = diff_lineups(team, current_players, expected)

            for change in changes:
                expected_name = change["expected"]
                actual_name = change.get("actual", "???")
                jersey = change["jersey_number"]

                # Look up impact of the missing player
                impact = 0.0
                if impact_df is not None:
                    impact = get_player_impact(
                        team, player_name=expected_name, impact_df=impact_df
                    )

                alert = {
                    "team": team,
                    "side": side,
                    "jersey": jersey,
                    "expected": expected_name,
                    "actual": actual_name,
                    "change_type": change["change_type"],
                    "impact": impact,
                }
                match_alerts.append(alert)

                # Accumulate adjustments
                if side == "home":
                    home_adj -= impact  # losing this player hurts home
                elif side == "away":
                    away_adj -= impact  # losing this player hurts away

        # Apply adjustments (cap per-team AND net total to ±0.15)
        MAX_TEAM_ADJ = 0.15
        MAX_NET_ADJ = 0.15
        home_adj = np.clip(home_adj, -MAX_TEAM_ADJ, MAX_TEAM_ADJ)
        away_adj = np.clip(away_adj, -MAX_TEAM_ADJ, MAX_TEAM_ADJ)
        total_adj = np.clip(home_adj - away_adj, -MAX_NET_ADJ, MAX_NET_ADJ)
        if abs(total_adj) > 0.001:
            old_prob = row["home_win_prob"]
            new_prob = np.clip(old_prob + total_adj, 0.05, 0.95)
            results.at[idx, "home_win_prob"] = new_prob
            results.at[idx, "away_win_prob"] = 1.0 - new_prob
            results.at[idx, "tip"] = home if new_prob >= 0.5 else away
            results.at[idx, "confidence"] = abs(new_prob - 0.5) * 2

        lineup_alerts.extend(match_alerts)

    # Print lineup report
    if lineup_alerts:
        print("  " + "=" * 65)
        print("  LINEUP CHANGES DETECTED")
        print("  " + "=" * 65)

        for alert in lineup_alerts:
            impact_str = ""
            if abs(alert["impact"]) > 0.001:
                impact_str = f" (impact: {alert['impact']:+.3f})"

            if alert["change_type"] == "REPLACED":
                print(f"    {alert['team']:30s} #{alert['jersey']:2d}: "
                      f"{alert['expected']} → {alert['actual']}{impact_str}")
            else:
                print(f"    {alert['team']:30s} #{alert['jersey']:2d}: "
                      f"{alert['expected']} → MISSING{impact_str}")

        # Summarise adjustments
        adjusted = results[results["home_win_prob"] != results.get("_orig_prob", results["home_win_prob"])]
        if lineup_alerts:
            print(f"\n    Total changes: {len(lineup_alerts)}")
    else:
        print("    No lineup changes detected vs expected starters")

    return results


# =====================================================================
# Output Formatting
# =====================================================================

def format_predictions(results: pd.DataFrame, round_num: int, year: int):
    """Print predictions as a tipping card with LOCK/LEAN/TOSS-UP categories.

    Strategy (based on 2025 data: favourite only won 62.4%):
      - LOCK (fav >= 65%): Always tip favourite
      - LEAN (fav 55-65%): Default favourite, flip if model strongly disagrees
      - TOSS-UP (fav < 55%): Use model prediction
      - MARGIN: from bookmaker spread or odds-implied
    """
    from tipping_advisor import (
        get_tip, print_tips, calculate_implied_prob,
    )

    tips = []
    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Get decimal odds
        h2h_home = row.get("h2h_home")
        h2h_away = row.get("h2h_away")

        # Fallback: convert from implied prob if no decimal odds
        if pd.isna(h2h_home) or pd.isna(h2h_away):
            odds_hp = row["odds_home_prob"]
            odds_ap = row["odds_away_prob"]
            overround = 1.05
            h2h_home = overround / odds_hp if odds_hp > 0 else 2.0
            h2h_away = overround / odds_ap if odds_ap > 0 else 2.0

        # Model prediction = home_win_prob from OptBlend
        model_pred = row["home_win_prob"]

        # Spread for margin
        spread = row.get("spread_home")
        spread = spread if pd.notna(spread) else None

        tip = get_tip(home, away, h2h_home, h2h_away,
                      model_pred=model_pred, spread=spread)
        tips.append(tip)

    print_tips(tips, round_num, year)


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
    df["_is_user_upcoming"] = True
    return df, detected_round


def main():
    parser = argparse.ArgumentParser(description="NRL Tipping Comp Predictor")
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
    parser.add_argument("--check-lineups", action="store_true",
                        help="Fetch NRL.com team lists and adjust for missing players")
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

            if args.check_lineups:
                results = check_lineups_and_adjust(results, round_num, year)

            if args.match:
                results = _filter_match(results, args.match)

            format_predictions(results, round_num, year)
            save_predictions(results, round_num, year)

            elapsed = time.time() - t_start
            print(f"\n  Completed in {elapsed:.0f}s (fast mode)")
            return

        # === FULL PATH: build features, train, predict, cache ===
        print("\n  STEP 2: Loading historical data")
        matches, ladders, odds, match_stats, player_match_stats = load_historical_data()

        print("\n  STEP 3: Elo parameters")
        elo_params = get_elo_params(matches, retune=args.retune_elo)

        print("\n  STEP 4: Feature engineering")
        historical, upcoming_feat, feature_cols = build_features(
            matches, ladders, odds, upcoming_api, elo_params, match_stats,
            player_match_stats
        )

        print("\n  STEP 5: Model training & prediction")
        results, artifacts = train_and_predict(historical, upcoming_feat, feature_cols)

        # Save cache for fast re-scoring
        save_model_cache(cp, artifacts, upcoming_feat, round_num, year)

        if args.check_lineups:
            results = check_lineups_and_adjust(results, round_num, year)

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
        matches, ladders, odds, match_stats, player_match_stats = load_historical_data()
        upcoming = load_upcoming_matches(csv_path, round_num, year)

        # Step 2: Get Elo parameters
        print("\n  STEP 2: Elo parameters")
        elo_params = get_elo_params(matches, retune=args.retune_elo)

        # Step 3: Build features
        print("\n  STEP 3: Feature engineering")
        historical, upcoming_feat, feature_cols = build_features(
            matches, ladders, odds, upcoming, elo_params, match_stats,
            player_match_stats
        )

        # Step 4: Train and predict
        print("\n  STEP 4: Model training & prediction")
        results, _artifacts = train_and_predict(historical, upcoming_feat, feature_cols)

        if args.check_lineups:
            results = check_lineups_and_adjust(results, round_num, year)

        if args.match:
            results = _filter_match(results, args.match)

        format_predictions(results, round_num, year)
        save_predictions(results, round_num, year)

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
