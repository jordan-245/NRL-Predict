"""
NRL Match Prediction - V4 Enhanced Pipeline
============================================
Builds on V3 with significant improvements across features, models, and ensembles.

NEW FEATURES (V4):
  1. Closing spread + spread movement (sharp money signal)
  2. Total line close + movement (expected scoring environment)
  3. Draw odds implied probability (competitiveness signal)
  4. Scoring consistency (rolling std dev of points scored/conceded)
  5. Close game tendency (% of games decided by <7 points)
  6. Points trend (recent scoring rate vs season average)
  7. Attendance-based home advantage proxy
  8. Kickoff time features (day game vs night game)
  9. Upset rate (how often team loses when favoured)
  10. Defensive efficiency features (points against trend)
  11. Lineup stability (players retained between games)
  12. Season stage interactions (early/mid/late season behavior)
  13. Odds-spread agreement (do H2H odds and spread tell same story)

NEW MODELS:
  - Random Forest (different bias-variance tradeoff)
  - Ridge Classifier (L2 regularized linear)
  - ExtraTrees (extra randomization)
  - Neural Network meta-learner (MLP stacking)

IMPROVED BLENDING:
  - Walk-forward OptBlend (no look-ahead in weight optimization)
  - Isotonic calibration before blending
  - More blend combos tested

Usage:
    python run_v4_pipeline.py
"""

import sys
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from scipy.optimize import minimize
import optuna
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING)

# Import V3 base functions
from pipelines import v3

# Walk-forward folds (same as V3)
FOLDS = v3.FOLDS

# V4-tuned hyperparameters (Optuna, 100 trials each on V4 194-feature set)
BEST_XGB_PARAMS = {
    'n_estimators': 348, 'max_depth': 5,
    'learning_rate': 0.007883713651576849,
    'subsample': 0.5787119138697776,
    'colsample_bytree': 0.23243000723775306,
    'reg_alpha': 0.006309024583131649,
    'reg_lambda': 0.23908624386240268,
    'min_child_weight': 12, 'gamma': 3.112173145335378,
    'eval_metric': 'logloss', 'verbosity': 0, 'random_state': 42,
}

BEST_LGB_PARAMS = {
    'n_estimators': 450, 'num_leaves': 44, 'max_depth': 2,
    'learning_rate': 0.011381907551166253,
    'subsample': 0.668591524465417,
    'colsample_bytree': 0.22840930494641537,
    'reg_alpha': 0.002996001252884443,
    'reg_lambda': 5.774885014717924e-07,
    'min_child_samples': 9,
    'random_state': 42, 'verbose': -1,
}

BEST_CAT_PARAMS = {
    'iterations': 472, 'depth': 6,
    'learning_rate': 0.010714785506587051,
    'l2_leaf_reg': 1.4525445616350277,
    'subsample': 0.6817498049854462,
    'colsample_bylevel': 0.658256433246225,
    'min_data_in_leaf': 19,
    'random_seed': 42, 'verbose': 0, 'allow_writing_files': False,
}

BEST_RF_PARAMS = {
    'n_estimators': 859, 'max_depth': 6,
    'min_samples_leaf': 25, 'max_features': 'sqrt',
    'random_state': 42, 'n_jobs': -1,
}

SAMPLE_WEIGHT_DECAY = 0.907

# V4 OptBlend weights.
#
# Simplified from the original 7-model blend which had fragile negative
# weights (LightGBM=-0.222, XGB_top50=-0.809).  Negative weights on
# correlated GBM variants overfit the walk-forward validation.
#
# CatBoost + Odds (Raw-Blend) is the most robust option from backtest:
#   68.1% accuracy, 0.5989 log loss — beats the old 7-model 67.6%.
# Training only CatBoost also halves the retrain time.
#
# Walk-forward backtest (2018-2025, 1538 games):
#   25% model = 68.7% (all-time best)
#   35% model = 68.5% (best on recent 2022-2025)
#   50% model = 68.3% (original)
#   0%  model = 68.1% (pure odds)
#
# Using 35% as compromise: near-optimal across eras, and model
# accuracy is improving as training data grows.
# Toss-up games (<55%): 30% model gives 60.3% (best tier result).
V4_BLEND_WEIGHTS = {
    "CAT_top50":  0.35,
}
V4_BLEND_ODDS_WEIGHT = 0.65  # 1 - sum(model weights)


def safe_log_loss(y_true, y_prob, eps=1e-7):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return log_loss(y_true, y_prob)


def compute_metrics(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": safe_log_loss(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob),
    }


# =========================================================================
# V4 NEW FEATURE FUNCTIONS
# =========================================================================

def compute_v4_odds_features(matches):
    """Compute additional odds features not in V3: spreads, totals, draw odds."""
    print("\n" + "=" * 80)
    print("  V4: COMPUTING ENHANCED ODDS FEATURES")
    print("=" * 80)

    df = matches.copy()

    # --- Closing spread and spread movement ---
    if "line_home_close" in df.columns:
        df["spread_home_close"] = pd.to_numeric(df["line_home_close"], errors="coerce")
    if "line_home_open" in df.columns and "line_home_close" in df.columns:
        open_sp = pd.to_numeric(df["line_home_open"], errors="coerce")
        close_sp = pd.to_numeric(df.get("line_home_close", df["line_home_open"]), errors="coerce")
        df["spread_movement"] = close_sp - open_sp
        df["spread_movement_abs"] = df["spread_movement"].abs()

    # --- Spread-odds agreement (do spread and H2H tell the same story?) ---
    if "odds_home_prob" in df.columns and "spread_home_open" in df.columns:
        # If home is H2H favourite (prob>0.5) AND spread is negative (home giving points), they agree
        odds_fav_home = df["odds_home_prob"] > 0.5
        spread_fav_home = pd.to_numeric(df["spread_home_open"], errors="coerce") < 0
        df["odds_spread_agree"] = (odds_fav_home == spread_fav_home).astype(float)
        # Magnitude of disagreement: how much do they differ in implied advantage
        df["odds_spread_disagree_mag"] = (
            df["odds_home_prob"] - 0.5
        ).abs() - (-pd.to_numeric(df["spread_home_open"], errors="coerce").fillna(0) / 20.0)

    # --- Total line features ---
    if "total_line_close" in df.columns:
        df["total_line_close"] = pd.to_numeric(df["total_line_close"], errors="coerce")
    if "total_line_open" in df.columns and "total_line_close" in df.columns:
        t_open = pd.to_numeric(df["total_line_open"], errors="coerce")
        t_close = pd.to_numeric(df["total_line_close"], errors="coerce")
        df["total_movement"] = t_close - t_open
        df["total_movement_abs"] = df["total_movement"].abs()

    # --- Draw odds / competitiveness signal ---
    if "h2h_draw" in df.columns:
        draw_odds = pd.to_numeric(df["h2h_draw"], errors="coerce")
        df["implied_draw_prob"] = 1.0 / draw_odds
        # Lower draw odds = market thinks teams are more evenly matched
        df["draw_competitiveness"] = df["implied_draw_prob"]  # Higher = more competitive

    # --- Odds range (closing) as market uncertainty ---
    if "h2h_home_max" in df.columns and "h2h_home_min" in df.columns:
        df["odds_home_range_close"] = (
            pd.to_numeric(df["h2h_home_max"], errors="coerce") -
            pd.to_numeric(df["h2h_home_min"], errors="coerce")
        )
    if "h2h_away_max" in df.columns and "h2h_away_min" in df.columns:
        df["odds_away_range_close"] = (
            pd.to_numeric(df["h2h_away_max"], errors="coerce") -
            pd.to_numeric(df["h2h_away_min"], errors="coerce")
        )

    # --- Overround as market confidence ---
    if "odds_overround" in df.columns:
        df["market_confidence"] = 1.0 / df["odds_overround"]  # Lower overround = more confident market

    n_new = sum(1 for c in ["spread_home_close", "spread_movement", "spread_movement_abs",
                             "odds_spread_agree", "odds_spread_disagree_mag",
                             "total_line_close", "total_movement", "total_movement_abs",
                             "implied_draw_prob", "draw_competitiveness",
                             "odds_home_range_close", "odds_away_range_close",
                             "market_confidence"] if c in df.columns)
    print(f"  Added {n_new} new V4 odds features")
    return df


def compute_scoring_consistency_features(matches):
    """Compute scoring consistency (rolling std), close game tendency, and points trends."""
    print("\n" + "=" * 80)
    print("  V4: COMPUTING SCORING CONSISTENCY & TREND FEATURES")
    print("=" * 80)

    df = matches.copy()
    windows = [5, 8]

    # Build team match log (similar to V3 but for additional stats)
    home_log = pd.DataFrame({
        "match_idx": range(len(df)), "team": df["home_team"],
        "points_for": df["home_score"], "points_against": df["away_score"],
        "date": df["date"],
    })
    away_log = pd.DataFrame({
        "match_idx": range(len(df)), "team": df["away_team"],
        "points_for": df["away_score"], "points_against": df["home_score"],
        "date": df["date"],
    })
    log = pd.concat([home_log, away_log], ignore_index=True)
    log["margin"] = log["points_for"] - log["points_against"]
    log["margin_abs"] = log["margin"].abs()
    log["close_game"] = (log["margin_abs"] <= 6).astype(float)
    log["win"] = (log["margin"] > 0).astype(float)
    log = log.sort_values(["team", "date", "match_idx"]).reset_index(drop=True)

    # Build lookup
    lookup = {}
    for team in log["team"].unique():
        t_log = log[log["team"] == team].reset_index(drop=True)
        for i, row in t_log.iterrows():
            midx = int(row["match_idx"])
            key = (team, midx)
            lookup.setdefault(key, {})

            prior = t_log.iloc[:i]
            if len(prior) < 3:
                for w in windows:
                    lookup[key][f"pf_std_{w}"] = np.nan
                    lookup[key][f"pa_std_{w}"] = np.nan
                    lookup[key][f"close_game_rate_{w}"] = np.nan
                    lookup[key][f"pf_trend_{w}"] = np.nan
                    lookup[key][f"pa_trend_{w}"] = np.nan
                lookup[key]["upset_rate_5"] = np.nan
                continue

            for w in windows:
                pw = prior.tail(w)
                # Scoring consistency (std dev of points scored/conceded)
                lookup[key][f"pf_std_{w}"] = pw["points_for"].std() if len(pw) >= 3 else np.nan
                lookup[key][f"pa_std_{w}"] = pw["points_against"].std() if len(pw) >= 3 else np.nan
                # Close game tendency
                lookup[key][f"close_game_rate_{w}"] = pw["close_game"].mean()
                # Points trend: last 3 vs last W (increasing or decreasing scoring)
                if len(pw) >= 3:
                    recent_3_pf = pw.tail(3)["points_for"].mean()
                    all_w_pf = pw["points_for"].mean()
                    lookup[key][f"pf_trend_{w}"] = recent_3_pf - all_w_pf
                    recent_3_pa = pw.tail(3)["points_against"].mean()
                    all_w_pa = pw["points_against"].mean()
                    lookup[key][f"pa_trend_{w}"] = recent_3_pa - all_w_pa
                else:
                    lookup[key][f"pf_trend_{w}"] = np.nan
                    lookup[key][f"pa_trend_{w}"] = np.nan

    # Attach to dataframe
    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        for w in windows:
            for stat in ("pf_std", "pa_std", "close_game_rate", "pf_trend", "pa_trend"):
                col_name = f"{side}_{stat}_{w}"
                df[col_name] = [
                    lookup.get((df.iloc[i][team_col], i), {}).get(f"{stat}_{w}", np.nan)
                    for i in range(len(df))
                ]

    # Differentials
    for w in windows:
        df[f"pf_std_diff_{w}"] = df[f"home_pf_std_{w}"] - df[f"away_pf_std_{w}"]
        df[f"close_game_diff_{w}"] = df[f"home_close_game_rate_{w}"] - df[f"away_close_game_rate_{w}"]
        df[f"pf_trend_diff_{w}"] = df[f"home_pf_trend_{w}"] - df[f"away_pf_trend_{w}"]
        df[f"pa_trend_diff_{w}"] = df[f"home_pa_trend_{w}"] - df[f"away_pa_trend_{w}"]

    # Scoring volatility ratio (high = unpredictable)
    for side in ["home", "away"]:
        pf_mean = df.get(f"{side}_avg_pf_5")
        pf_std = df.get(f"{side}_pf_std_5")
        if pf_mean is not None and pf_std is not None:
            df[f"{side}_scoring_cv_5"] = pf_std / pf_mean.clip(lower=1)

    if "home_scoring_cv_5" in df.columns:
        df["scoring_cv_diff_5"] = df["home_scoring_cv_5"] - df["away_scoring_cv_5"]

    n_cols = sum(1 for c in df.columns if any(p in c for p in
        ["pf_std_", "pa_std_", "close_game_", "pf_trend_", "pa_trend_", "scoring_cv_"]))
    print(f"  Added {n_cols} scoring consistency & trend features")
    return df


def compute_attendance_features(matches):
    """Compute attendance-based home advantage features."""
    print("\n" + "=" * 80)
    print("  V4: COMPUTING ATTENDANCE FEATURES")
    print("=" * 80)

    df = matches.copy()

    if "attendance" not in df.columns:
        print("  No attendance data available, skipping")
        return df

    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

    # Build per-team average attendance
    team_attend = {}
    team_attend_count = {}

    attend_norm_list = []
    attend_high_list = []

    for _, row in df.iterrows():
        home = row.get("home_team", "")
        att = row.get("attendance")

        if pd.isna(att) or not home:
            attend_norm_list.append(np.nan)
            attend_high_list.append(np.nan)
            continue

        # Pre-match: use accumulated average
        avg_att = team_attend.get(home, 0)
        cnt = team_attend_count.get(home, 0)

        if cnt >= 3:
            team_avg = avg_att / cnt
            attend_norm_list.append(att / team_avg)
            attend_high_list.append(1.0 if att > team_avg * 1.2 else 0.0)
        else:
            attend_norm_list.append(np.nan)
            attend_high_list.append(np.nan)

        # Update accumulator
        team_attend[home] = team_attend.get(home, 0) + att
        team_attend_count[home] = team_attend_count.get(home, 0) + 1

    df["attendance_normalized"] = attend_norm_list
    df["attendance_high"] = attend_high_list

    n_valid = df["attendance_normalized"].notna().sum()
    print(f"  Added attendance features ({n_valid}/{len(df)} valid)")
    return df


def compute_kickoff_features(matches):
    """Compute kickoff time features (day vs night, early vs late)."""
    print("\n" + "=" * 80)
    print("  V4: COMPUTING KICKOFF TIME FEATURES")
    print("=" * 80)

    df = matches.copy()

    if "kickoff_time" not in df.columns and "date" not in df.columns:
        print("  No kickoff time data available, skipping")
        return df

    # Try to extract hour from kickoff_time or date
    kickoff_hour = []
    for _, row in df.iterrows():
        kt = row.get("kickoff_time", "")
        dt = row.get("date")
        hour = np.nan

        if pd.notna(kt) and str(kt).strip():
            try:
                # Parse "8:00 PM" style
                t_str = str(kt).strip().upper()
                if "PM" in t_str or "AM" in t_str:
                    parts = t_str.replace("PM", "").replace("AM", "").strip().split(":")
                    h = int(parts[0])
                    if "PM" in t_str and h != 12:
                        h += 12
                    elif "AM" in t_str and h == 12:
                        h = 0
                    hour = h
            except (ValueError, IndexError):
                pass

        if np.isnan(hour) and pd.notna(dt):
            try:
                hour = pd.to_datetime(dt).hour
            except Exception:
                pass

        kickoff_hour.append(hour)

    df["kickoff_hour"] = kickoff_hour
    df["is_night_game"] = (df["kickoff_hour"] >= 18).astype(float)
    df["is_afternoon_game"] = ((df["kickoff_hour"] >= 13) & (df["kickoff_hour"] < 18)).astype(float)
    df["is_day_game"] = (df["kickoff_hour"] < 13).astype(float)

    n_valid = df["kickoff_hour"].notna().sum()
    print(f"  Added kickoff time features ({n_valid}/{len(df)} valid)")
    return df


def compute_lineup_stability_features(matches):
    """Compute lineup stability from player_appearances.parquet.

    Uses the actual player appearance data rather than trying to parse list
    columns from matches (which were dropped during Parquet serialisation).
    """
    print("\n" + "=" * 80)
    print("  V4: COMPUTING LINEUP STABILITY FEATURES")
    print("=" * 80)

    df = matches.copy()

    # Try loading player appearances for proper stability computation
    app_path = PROJECT_ROOT / "data" / "processed" / "player_appearances.parquet"
    if app_path.exists():
        appearances = pd.read_parquet(app_path)
        starters = appearances[appearances["is_starter"]].copy()

        # Build per-team per-match starter sets
        match_starters = (
            starters.groupby(["match_id", "team"])["player_name"]
            .apply(set).to_dict()
        )

        # Build team previous lineup tracker
        team_prev = {}
        stability_vals = {"home": [], "away": []}

        for _, row in df.iterrows():
            yr = row.get("year", 0)
            rnd = row.get("round", "")

            for side, team_col in [("home", "home_team"), ("away", "away_team")]:
                team = row.get(team_col, "")
                # Build match_id matching the format from build_player_data
                opp_col = "away_team" if side == "home" else "home_team"
                opp = row.get(opp_col, "")
                mid = f"{yr}_r{rnd}_{row.get('home_team', '')}_v_{row.get('away_team', '')}"

                current = match_starters.get((mid, team))
                if current is None:
                    stability_vals[side].append(np.nan)
                    continue

                prev = team_prev.get(team)
                if prev and len(current) > 0 and len(prev) > 0:
                    retained = len(current & prev)
                    stability = retained / max(len(current), 1)
                    stability_vals[side].append(stability)
                else:
                    stability_vals[side].append(np.nan)

                if len(current) > 0:
                    team_prev[team] = current

        df["home_lineup_stability"] = stability_vals["home"]
        df["away_lineup_stability"] = stability_vals["away"]
        df["lineup_stability_diff"] = (
            pd.to_numeric(pd.Series(stability_vals["home"]), errors="coerce") -
            pd.to_numeric(pd.Series(stability_vals["away"]), errors="coerce")
        ).values

        n_valid = df["home_lineup_stability"].notna().sum()
        print(f"  Added lineup stability features from player_appearances ({n_valid}/{len(df)} valid)")
    else:
        # Fallback: no lineup data available
        print("  No player_appearances.parquet found — lineup stability will be NaN")
        for col in ["home_lineup_stability", "away_lineup_stability", "lineup_stability_diff"]:
            df[col] = np.nan

    return df


def compute_player_impact_features(matches):
    """Compute player impact features from player_impact.parquet.

    Adds per-match features:
    - home/away_spine_impact_sum: total impact of starting spine players
    - spine_impact_diff: difference between home and away spine impact
    - home/away_top_absent_impact: impact of highest-impact absent players
    """
    print("\n" + "=" * 80)
    print("  V4: COMPUTING PLAYER IMPACT FEATURES")
    print("=" * 80)

    df = matches.copy()

    impact_path = PROJECT_ROOT / "data" / "processed" / "player_impact.parquet"
    app_path = PROJECT_ROOT / "data" / "processed" / "player_appearances.parquet"

    if not impact_path.exists() or not app_path.exists():
        print("  No player impact data available, skipping")
        for col in ["home_spine_impact", "away_spine_impact", "spine_impact_diff",
                     "home_total_impact", "away_total_impact", "total_impact_diff"]:
            df[col] = np.nan
        return df

    impact = pd.read_parquet(impact_path)
    appearances = pd.read_parquet(app_path)

    # Build lookup: (team, player_name) → weighted_impact
    impact_lookup = {}
    spine_lookup = {}
    for _, row in impact.iterrows():
        key = (row["team"], row["player_name"])
        impact_lookup[key] = row["weighted_impact"]
        if row["is_spine"]:
            spine_lookup[key] = row["weighted_impact"]

    # Build per-match starter sets
    starters = appearances[appearances["is_starter"]].copy()
    spine_starters = starters[starters["is_spine"]].copy()

    match_starters_set = (
        starters.groupby(["match_id", "team"])["player_name"]
        .apply(set).to_dict()
    )
    match_spine_set = (
        spine_starters.groupby(["match_id", "team"])["player_name"]
        .apply(set).to_dict()
    )

    # Compute features per match
    home_spine = []
    away_spine = []
    home_total = []
    away_total = []

    for _, row in df.iterrows():
        yr = row.get("year", 0)
        rnd = row.get("round", "")
        ht = row.get("home_team", "")
        at = row.get("away_team", "")
        mid = f"{yr}_r{rnd}_{ht}_v_{at}"

        for side, team, s_list, t_list in [
            ("home", ht, home_spine, home_total),
            ("away", at, away_spine, away_total),
        ]:
            # Get spine starters for this match
            spine_players = match_spine_set.get((mid, team), set())
            all_starters = match_starters_set.get((mid, team), set())

            if not all_starters:
                s_list.append(np.nan)
                t_list.append(np.nan)
                continue

            # Sum impacts for spine players
            spine_sum = sum(
                impact_lookup.get((team, p), 0.0) for p in spine_players
            )
            s_list.append(spine_sum)

            # Sum impacts for all starters
            total_sum = sum(
                impact_lookup.get((team, p), 0.0) for p in all_starters
            )
            t_list.append(total_sum)

    df["home_spine_impact"] = home_spine
    df["away_spine_impact"] = away_spine
    df["spine_impact_diff"] = (
        pd.to_numeric(pd.Series(home_spine), errors="coerce") -
        pd.to_numeric(pd.Series(away_spine), errors="coerce")
    ).values

    df["home_total_impact"] = home_total
    df["away_total_impact"] = away_total
    df["total_impact_diff"] = (
        pd.to_numeric(pd.Series(home_total), errors="coerce") -
        pd.to_numeric(pd.Series(away_total), errors="coerce")
    ).values

    n_valid = df["home_spine_impact"].notna().sum()
    print(f"  Added player impact features ({n_valid}/{len(df)} valid)")
    return df


def compute_v4_engineered_features(df):
    """Compute V4 interaction and derived features."""
    print("\n" + "=" * 80)
    print("  V4: COMPUTING ENGINEERED INTERACTION FEATURES")
    print("=" * 80)

    df = df.copy()

    # --- Season stage (early/mid/late) ---
    rn = pd.to_numeric(df.get("round_number"), errors="coerce")
    df["is_early_season"] = (rn <= 5).astype(float)
    df["is_mid_season"] = ((rn > 5) & (rn <= 18)).astype(float)
    df["is_late_season"] = (rn > 18).astype(float)

    # --- Elo x season stage interactions ---
    if "elo_diff" in df.columns:
        df["elo_diff_x_late"] = df["elo_diff"] * df["is_late_season"]
        df["elo_diff_x_early"] = df["elo_diff"] * df["is_early_season"]

    # --- Form x season stage ---
    if "win_rate_diff_5" in df.columns:
        df["form_x_late"] = df["win_rate_diff_5"] * df["is_late_season"]

    # --- Defensive efficiency trend ---
    if "home_pa_trend_5" in df.columns:
        # Negative pa_trend = defense improving (conceding fewer points recently)
        df["home_defense_improving"] = (-df["home_pa_trend_5"]).clip(lower=0)
        df["away_defense_improving"] = (-df["away_pa_trend_5"]).clip(lower=0)
        df["defense_trend_diff"] = df["home_defense_improving"] - df["away_defense_improving"]

    # --- Scoring environment interaction ---
    if "total_line_open" in df.columns and "home_avg_pf_5" in df.columns:
        expected_total = pd.to_numeric(df["total_line_open"], errors="coerce")
        actual_scoring = df["home_avg_pf_5"] + df["away_avg_pf_5"]
        df["scoring_env_ratio"] = actual_scoring / expected_total.clip(lower=20)

    # --- Consistency x favourite interaction ---
    if "home_pf_std_5" in df.columns and "odds_home_prob" in df.columns:
        # Consistent favourites are more likely to win
        df["fav_consistency"] = (
            df["odds_home_prob"] * (1 - df["home_pf_std_5"].fillna(10) / 20.0)
        )

    # --- Spread x Elo agreement ---
    if "spread_home_open" in df.columns and "elo_diff" in df.columns:
        spread_val = pd.to_numeric(df["spread_home_open"], errors="coerce")
        elo_val = df["elo_diff"]
        # Both negative spread (home giving points) and positive elo_diff agree
        df["elo_spread_agree"] = (
            (spread_val < 0) & (elo_val > 0) |
            (spread_val > 0) & (elo_val < 0)
        ).astype(float)

    # --- Rest advantage when team is strong ---
    if "rest_diff" in df.columns and "elo_diff" in df.columns:
        df["strong_team_rested"] = df["elo_diff"] * df["rest_diff"].clip(lower=0) / 7.0

    # --- Home ground advantage multiplied by form ---
    if "is_home_ground" in df.columns and "home_form_momentum" in df.columns:
        df["home_ground_x_form"] = (
            df["is_home_ground"].fillna(0) * df["home_form_momentum"].fillna(0)
        )

    n_new = sum(1 for c in ["is_early_season", "is_mid_season", "is_late_season",
                             "elo_diff_x_late", "elo_diff_x_early", "form_x_late",
                             "home_defense_improving", "away_defense_improving",
                             "defense_trend_diff", "scoring_env_ratio",
                             "fav_consistency", "elo_spread_agree",
                             "strong_team_rested", "home_ground_x_form"] if c in df.columns)
    print(f"  Added {n_new} V4 engineered features")
    return df


# =========================================================================
# TEAM SEASON STATS FEATURES
# =========================================================================
# SUPERCOACH MATCHUP FEATURES
# =========================================================================

SC_POINTS_ALLOWED_PATH = PROJECT_ROOT / "data" / "processed" / "sc_points_allowed.parquet"

# Position groups for aggregation
_SC_SPINE = {"FLB", "HFB", "FE", "HOK"}
_SC_FORWARDS = {"FRF", "2RF"}
_SC_BACKS = {"CTW"}


def compute_sc_matchup_features(matches):
    """Add SuperCoach defensive-matchup features using prior-season data.

    For a match in year Y, each team's *opponent* gets their SC points-allowed
    profile from year Y-1.  This measures "how leaky is the opponent's defence
    at each position group?" — a matchup feature, not a quality metric.

    Features added (12 total):
      - home_opp_sc_spine / away_opp_sc_spine / sc_spine_diff
        (avg SC points away/home team allowed to spine positions last season)
      - home_opp_sc_forward / away_opp_sc_forward / sc_forward_diff
      - home_opp_sc_back / away_opp_sc_back / sc_back_diff
      - home_opp_sc_total / away_opp_sc_total / sc_total_diff
    """
    print("\n" + "=" * 80)
    print("  V4: COMPUTING SUPERCOACH MATCHUP FEATURES")
    print("=" * 80)

    df = matches.copy()

    if not SC_POINTS_ALLOWED_PATH.exists():
        print("  No sc_points_allowed.parquet found — skipping")
        for grp in ["spine", "forward", "back", "total"]:
            df[f"home_opp_sc_{grp}"] = np.nan
            df[f"away_opp_sc_{grp}"] = np.nan
            df[f"sc_{grp}_diff"] = np.nan
        return df

    sc = pd.read_parquet(SC_POINTS_ALLOWED_PATH)

    # Build lookup: (team, season) → {group: avg_points_allowed}
    # Pre-aggregate to position groups
    sc_lookup: dict[tuple, dict] = {}
    for (team, season), grp in sc.groupby(["team", "season"]):
        pos_vals = dict(zip(grp["position"], grp["avg_points_allowed"]))

        spine_vals = [v for p, v in pos_vals.items() if p in _SC_SPINE]
        fwd_vals = [v for p, v in pos_vals.items() if p in _SC_FORWARDS]
        back_vals = [v for p, v in pos_vals.items() if p in _SC_BACKS]
        all_vals = list(pos_vals.values())

        sc_lookup[(team, season)] = {
            "spine": float(np.mean(spine_vals)) if spine_vals else np.nan,
            "forward": float(np.mean(fwd_vals)) if fwd_vals else np.nan,
            "back": float(np.mean(back_vals)) if back_vals else np.nan,
            "total": float(np.mean(all_vals)) if all_vals else np.nan,
        }

    # For each match in year Y:
    #   home team faces away_team — use away_team's Y-1 allowed as "home_opp_sc_*"
    #   away team faces home_team — use home_team's Y-1 allowed as "away_opp_sc_*"
    for grp in ["spine", "forward", "back", "total"]:
        home_vals = []
        away_vals = []
        for _, row in df.iterrows():
            year = row.get("year", row.get("season"))
            prior = int(year) - 1 if pd.notna(year) else None

            # Home team's opponent (away_team) defensive profile from prior season
            away_profile = sc_lookup.get((row["away_team"], prior), {})
            home_vals.append(away_profile.get(grp, np.nan))

            # Away team's opponent (home_team) defensive profile from prior season
            home_profile = sc_lookup.get((row["home_team"], prior), {})
            away_vals.append(home_profile.get(grp, np.nan))

        df[f"home_opp_sc_{grp}"] = home_vals
        df[f"away_opp_sc_{grp}"] = away_vals
        df[f"sc_{grp}_diff"] = df[f"home_opp_sc_{grp}"] - df[f"away_opp_sc_{grp}"]

    # Coverage stats
    coverage = df["home_opp_sc_spine"].notna().mean() * 100
    n_features = 12
    print(f"  Added {n_features} SC matchup features | Coverage: {coverage:.0f}%")

    # Show sample values for sanity check
    valid = df[df["home_opp_sc_spine"].notna()]
    if len(valid) > 0:
        print(f"  SC spine allowed range: [{valid['home_opp_sc_spine'].min():.3f}, "
              f"{valid['home_opp_sc_spine'].max():.3f}]")
        print(f"  SC total diff range: [{valid['sc_total_diff'].min():.3f}, "
              f"{valid['sc_total_diff'].max():.3f}]")

    return df


# =========================================================================
# REFEREE FEATURES
# =========================================================================

OFFICIALS_PATH = PROJECT_ROOT / "data" / "processed" / "match_officials.parquet"


def compute_referee_features(matches):
    """Add referee-based features: rolling home win rate by ref.

    Uses only historical data (look-back) so no leakage.
    """
    print("\n" + "=" * 80)
    print("  V4: COMPUTING REFEREE FEATURES")
    print("=" * 80)

    df = matches.copy()

    if not OFFICIALS_PATH.exists():
        print("  No match_officials.parquet found — skipping")
        for c in ["ref_home_win_rate", "ref_games", "ref_is_high_home"]:
            df[c] = np.nan
        return df

    officials = pd.read_parquet(OFFICIALS_PATH)
    officials = officials[officials["referee"] != ""].copy()

    # Build referee → match lookup
    off_lookup = {}
    for _, row in officials.iterrows():
        key = (row["year"], str(row["round"]), row["home_team"], row["away_team"])
        off_lookup[key] = row["referee"]

    # Assign referee to each match
    refs = []
    for _, row in df.iterrows():
        key = (row.get("year"), str(row.get("round", "")), row.get("home_team"), row.get("away_team"))
        refs.append(off_lookup.get(key, ""))
    df["_referee"] = refs

    # Compute rolling referee home win rate (look-back only)
    ref_home_wins = {}  # ref → [list of (year, round, home_win)]
    ref_hw_rate = []
    ref_game_count = []

    for _, row in df.iterrows():
        ref = row["_referee"]
        if ref and ref in ref_home_wins:
            history = ref_home_wins[ref]
            # Use all prior games by this ref
            if history:
                rate = np.mean(history)
                ref_hw_rate.append(rate)
                ref_game_count.append(len(history))
            else:
                ref_hw_rate.append(np.nan)
                ref_game_count.append(0)
        else:
            ref_hw_rate.append(np.nan)
            ref_game_count.append(0)

        # Update history (after computing feature — no leakage)
        if ref and pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
            hw = 1.0 if row["home_score"] > row["away_score"] else 0.0
            if ref not in ref_home_wins:
                ref_home_wins[ref] = []
            ref_home_wins[ref].append(hw)

    df["ref_home_win_rate"] = ref_hw_rate
    df["ref_games"] = ref_game_count
    # Binary: is this ref in the top quartile for home bias?
    df["ref_is_high_home"] = (df["ref_home_win_rate"] > 0.60).astype(float)

    n_valid = df["ref_home_win_rate"].notna().sum()
    n_refs = df["_referee"].nunique() - (1 if "" in df["_referee"].values else 0)
    df = df.drop(columns=["_referee"])
    print(f"  Added referee features ({n_valid}/{len(df)} valid, {n_refs} unique refs)")
    return df


# =========================================================================

TEAM_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "team_season_stats.parquet"

# Core per-game averages to use as features
_TS_COLS = [
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
]


def compute_team_stats_features(matches):
    """Merge prior-season team stats onto match rows.

    For a match in year Y, each team gets its stats from year Y-1.
    Creates home_ts_*, away_ts_*, and ts_diff_* columns.
    """
    print("\n" + "=" * 80)
    print("  V4: COMPUTING TEAM SEASON STATS FEATURES")
    print("=" * 80)

    df = matches.copy()

    if not TEAM_STATS_PATH.exists():
        print("  No team_season_stats.parquet found — skipping")
        return df

    ts = pd.read_parquet(TEAM_STATS_PATH)
    available = [c for c in _TS_COLS if c in ts.columns]
    if not available:
        print("  No matching stat columns found — skipping")
        return df

    # Home team: use year Y-1 stats for year Y matches
    home_merge = ts[["year", "team"] + available].copy()
    home_merge["merge_year"] = home_merge["year"] + 1
    home_merge = home_merge.drop(columns=["year"])
    home_merge = home_merge.rename(columns={"team": "home_team", "merge_year": "year"})
    home_merge = home_merge.rename(columns={c: f"home_ts_{c}" for c in available})

    # Away team: same logic
    away_merge = ts[["year", "team"] + available].copy()
    away_merge["merge_year"] = away_merge["year"] + 1
    away_merge = away_merge.drop(columns=["year"])
    away_merge = away_merge.rename(columns={"team": "away_team", "merge_year": "year"})
    away_merge = away_merge.rename(columns={c: f"away_ts_{c}" for c in available})

    pre_len = len(df)
    df = df.merge(home_merge, on=["year", "home_team"], how="left")
    df = df.merge(away_merge, on=["year", "away_team"], how="left")
    assert len(df) == pre_len, f"Merge changed row count: {pre_len} → {len(df)}"

    # Diff features
    for stat in available:
        h_col = f"home_ts_{stat}"
        a_col = f"away_ts_{stat}"
        df[f"ts_diff_{stat}"] = df[h_col] - df[a_col]

    n_new = len(available) * 3  # home + away + diff
    coverage = df[f"home_ts_{available[0]}"].notna().mean() * 100
    print(f"  Added {n_new} team stats features ({coverage:.0f}% coverage)")
    return df


# =========================================================================
# ROLLING PER-GAME MATCH STATS FEATURES
# =========================================================================

ROLLING_MATCH_STATS = [
    "completion_rate", "line_breaks", "tackle_breaks", "errors", "missed_tackles",
    "all_run_metres", "possession_pct", "effective_tackle_pct", "post_contact_metres", "offloads",
]
ROLLING_WINDOWS = [3, 5]


def compute_rolling_match_stats_features(matches, match_stats_df):
    """Compute rolling per-game match stats features (process quality) for each team.

    For each team in each match, computes rolling averages of process quality
    stats (completion rate, line breaks, tackle breaks, etc.) over the last
    3 and 5 games, using ONLY prior matches (no look-ahead bias).

    Adds columns:
      - home_ms_{stat}_{window}   — home team rolling average
      - away_ms_{stat}_{window}   — away team rolling average
      - ms_diff_{stat}_{window}   — home minus away differential

    Total new features: 10 stats × 2 windows × 3 (home/away/diff) = 60.

    Parameters
    ----------
    matches : pd.DataFrame
        Main matches DataFrame (already sorted chronologically, reset index).
    match_stats_df : pd.DataFrame or None
        Per-game team stats from match_stats.parquet.  Schema has columns:
        year, round, home_team, away_team, home_{stat}, away_{stat}, ...

    Returns
    -------
    pd.DataFrame
        matches with new rolling match stats columns appended.
    """
    print("\n" + "=" * 80)
    print("  V4: COMPUTING ROLLING MATCH STATS FEATURES")
    print("=" * 80)

    if match_stats_df is None or len(match_stats_df) == 0:
        print("  WARNING: match_stats_df is None or empty — skipping rolling match stats features")
        return matches

    df = matches.copy().reset_index(drop=True)
    ms = match_stats_df.copy()

    # ── Identify available stats ─────────────────────────────────────────────
    available_stats = [
        s for s in ROLLING_MATCH_STATS
        if f"home_{s}" in ms.columns and f"away_{s}" in ms.columns
    ]
    if not available_stats:
        print("  WARNING: No matching stat columns in match_stats_df — skipping")
        return matches

    # ── Normalise join keys ───────────────────────────────────────────────────
    ms["year"] = pd.to_numeric(ms["year"], errors="coerce").astype("Int64")
    ms["_round_str"] = ms["round"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["_round_str"] = df["round"].astype(str)

    # Add positional index for lookup (matches is sorted chronologically)
    df["_match_idx"] = range(len(df))

    # ── Join match_stats with matches to get date + match_idx ─────────────────
    stat_home_cols = [f"home_{s}" for s in available_stats]
    stat_away_cols = [f"away_{s}" for s in available_stats]
    ms_slim = ms[["year", "_round_str", "home_team", "away_team"] + stat_home_cols + stat_away_cols].copy()

    merge_ref = df[["_match_idx", "year", "_round_str", "home_team", "away_team", "date"]].copy()

    joined = ms_slim.merge(
        merge_ref,
        on=["year", "_round_str", "home_team", "away_team"],
        how="inner",
    )

    if len(joined) == 0:
        print("  WARNING: No matches joined between match_stats and matches "
              "— check team name standardisation")
        df = df.drop(columns=["_match_idx", "_round_str"], errors="ignore")
        return df

    print(f"  Joined {len(joined)} match-stat rows to matches")

    # ── Build per-team match log ───────────────────────────────────────────────
    # Each match_stats row → two entries (one per team, stats from their perspective)
    home_records = []
    away_records = []
    for _, row in joined.iterrows():
        base = {"match_idx": int(row["_match_idx"]), "date": row["date"]}
        h = dict(base, team=row["home_team"])
        a = dict(base, team=row["away_team"])
        for stat in available_stats:
            h[stat] = row.get(f"home_{stat}", np.nan)
            a[stat] = row.get(f"away_{stat}", np.nan)
        home_records.append(h)
        away_records.append(a)

    team_log = pd.concat(
        [pd.DataFrame(home_records), pd.DataFrame(away_records)],
        ignore_index=True,
    )
    team_log = team_log.sort_values(["team", "date", "match_idx"]).reset_index(drop=True)

    # ── Build lookup: (team, match_idx) → {stat_window: value} ───────────────
    lookup: dict[tuple, dict] = {}
    for team in team_log["team"].unique():
        t_log = team_log[team_log["team"] == team].reset_index(drop=True)
        for i, row in t_log.iterrows():
            midx = int(row["match_idx"])
            key = (team, midx)
            lookup.setdefault(key, {})
            prior = t_log.iloc[:i]
            for w in ROLLING_WINDOWS:
                pw = prior.tail(w)
                for stat in available_stats:
                    vals = pw[stat].dropna()
                    lookup[key][f"{stat}_{w}"] = float(vals.mean()) if len(vals) > 0 else np.nan

    # ── Attach features to DataFrame ──────────────────────────────────────────
    n_added = 0
    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        for w in ROLLING_WINDOWS:
            for stat in available_stats:
                col_name = f"{side}_ms_{stat}_{w}"
                df[col_name] = [
                    lookup.get((df.at[i, team_col], i), {}).get(f"{stat}_{w}", np.nan)
                    for i in range(len(df))
                ]
                n_added += 1

    # Differentials (home − away)
    for w in ROLLING_WINDOWS:
        for stat in available_stats:
            h_col = f"home_ms_{stat}_{w}"
            a_col = f"away_ms_{stat}_{w}"
            df[f"ms_diff_{stat}_{w}"] = df[h_col] - df[a_col]
            n_added += 1

    # Summarise
    first_stat = available_stats[0]
    coverage = df[f"home_ms_{first_stat}_3"].notna().mean() * 100
    print(f"  Added {n_added} rolling match stats features "
          f"({len(available_stats)} stats × {len(ROLLING_WINDOWS)} windows × 3) "
          f"| Coverage: {coverage:.0f}%")

    # Clean up temporary columns
    df = df.drop(columns=["_match_idx", "_round_str"], errors="ignore")
    return df


# =========================================================================
# PLAYER FORM FEATURES
# =========================================================================

# Mapping from raw API stat field names → short names used in features
_SPINE_STAT_MAP: dict[str, str] = {
    "allRunMetres":       "run_metres",
    "lineBreaks":         "line_breaks",
    "tackleBreaks":       "tackle_breaks",
    "tryAssists":         "try_assists",
}
_SQUAD_STAT_MAP: dict[str, str] = {
    "fantasyPointsTotal": "fantasy",
    "minutesPlayed":      "minutes",
}
# Combined map for building the player log
_ALL_PLAYER_STAT_MAP: dict[str, str] = {**_SPINE_STAT_MAP, **_SQUAD_STAT_MAP}

# Jersey numbers that define the spine
_SPINE_JERSEYS: frozenset[int] = frozenset({1, 6, 7, 9})

# Rolling windows
_PLAYER_ROLLING_WINDOWS: list[int] = [3, 5]


def compute_player_form_features(matches, player_stats_df):
    """Compute per-player rolling form features aggregated to match level.

    For each match, identifies the starting spine (jerseys 1, 6, 7, 9) and
    starting 13, then computes rolling 3/5-game averages per player using
    ONLY prior games (no look-ahead bias).  Averages are then aggregated to
    team-level features.

    Adds columns (42 total):

    **Spine form (24)** — average of 4 spine players' rolling stats:
      - ``{home,away}_spine_run_metres_{3,5}``
      - ``{home,away}_spine_line_breaks_{3,5}``
      - ``{home,away}_spine_tackle_breaks_{3,5}``
      - ``{home,away}_spine_try_assists_{3,5}``
      - ``spine_diff_{stat}_{window}`` — home minus away (8 features)

    **Squad quality (12)** — average of starting 13's rolling stats:
      - ``{home,away}_squad_fantasy_{3,5}``
      - ``{home,away}_squad_minutes_{3,5}``
      - ``squad_diff_{stat}_{window}`` — home minus away (4 features)

    **Disruption (6)** — player changes vs previous game:
      - ``{home,away}_spine_changes``   — # spine positions with new player
      - ``spine_changes_diff``
      - ``{home,away}_squad_turnover``  — # starting-13 slots with new player
      - ``squad_turnover_diff``

    Parameters
    ----------
    matches : pd.DataFrame
        Main matches DataFrame, sorted chronologically, index reset.
    player_stats_df : pd.DataFrame or None
        Per-game player stats from ``player_match_stats.parquet``.
        Must have columns: year, round, team, player_id, jersey_number,
        is_starter, is_spine, plus the raw stat fields.

    Returns
    -------
    pd.DataFrame
        ``matches`` with new player form columns appended.
    """
    print("\n" + "=" * 80)
    print("  V4: COMPUTING PLAYER FORM FEATURES")
    print("=" * 80)

    if player_stats_df is None or len(player_stats_df) == 0:
        print("  WARNING: player_stats_df is None or empty — skipping player form features")
        return matches

    # Check required columns exist in player_stats_df
    required_cols = {"year", "round", "team", "player_id", "jersey_number",
                     "is_starter", "is_spine"}
    missing_cols = required_cols - set(player_stats_df.columns)
    if missing_cols:
        print(f"  WARNING: player_stats_df missing columns {missing_cols} — skipping")
        return matches

    # Check which stat columns are available
    available_spine_stats = {
        api: short for api, short in _SPINE_STAT_MAP.items()
        if api in player_stats_df.columns
    }
    available_squad_stats = {
        api: short for api, short in _SQUAD_STAT_MAP.items()
        if api in player_stats_df.columns
    }
    available_all_stats = {**available_spine_stats, **available_squad_stats}

    if not available_all_stats:
        print("  WARNING: No matching stat columns in player_stats_df — skipping")
        return matches

    df = matches.copy().reset_index(drop=True)
    df["_match_idx"] = range(len(df))
    ps = player_stats_df.copy()

    # ── Normalise join keys ───────────────────────────────────────────────────
    ps["year"] = pd.to_numeric(ps["year"], errors="coerce").astype("Int64")
    ps["_round_str"] = ps["round"].astype(str)
    ps["jersey_number"] = pd.to_numeric(ps["jersey_number"], errors="coerce").fillna(0).astype(int)
    ps["player_id"] = pd.to_numeric(ps["player_id"], errors="coerce")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["_round_str"] = df["round"].astype(str)

    # ── Build match reference: (year, round, team) → (date, match_idx) ───────
    # Each match creates two entries — one per team
    match_refs: list[dict] = []
    for i, row in df.iterrows():
        for team_col in ("home_team", "away_team"):
            match_refs.append({
                "year":       row["year"],
                "_round_str": row["_round_str"],
                "team":       row[team_col],
                "date":       row.get("date"),
                "match_idx":  int(row["_match_idx"]),
            })
    match_ref_df = pd.DataFrame(match_refs)
    match_ref_df["year"] = match_ref_df["year"].astype("Int64")

    # ── Rename raw stat fields → short names in a working copy ───────────────
    ps_slim = ps[
        ["year", "_round_str", "team", "player_id", "jersey_number",
         "is_starter", "is_spine"]
        + list(available_all_stats.keys())
    ].copy()
    for api_name, short_name in available_all_stats.items():
        ps_slim[short_name] = pd.to_numeric(ps_slim[api_name], errors="coerce")
    # Keep only short names (drop originals to avoid duplication)
    short_cols = list(available_all_stats.values())
    ps_slim = ps_slim.drop(columns=list(available_all_stats.keys()), errors="ignore")

    # ── Join player stats with match reference ────────────────────────────────
    ps_joined = ps_slim.merge(
        match_ref_df,
        on=["year", "_round_str", "team"],
        how="inner",
    )

    if len(ps_joined) == 0:
        print("  WARNING: No player stats matched to matches — check team names / round formats")
        df = df.drop(columns=["_match_idx", "_round_str"], errors="ignore")
        return df

    n_matches_joined = ps_joined["match_idx"].nunique()
    print(f"  Joined {len(ps_joined)} player-game rows across {n_matches_joined} matches")

    # ── Sort player log chronologically per player ────────────────────────────
    ps_joined = ps_joined.sort_values(
        ["player_id", "date", "match_idx"]
    ).reset_index(drop=True)

    # ── Build per-player rolling-average lookup ───────────────────────────────
    # lookup[(player_id, match_idx)] = {short_name_w: value, ...}
    lookup: dict[tuple, dict] = {}

    for pid, p_group in ps_joined.groupby("player_id", sort=False):
        p_log = p_group.reset_index(drop=True)
        for i in range(len(p_log)):
            midx = int(p_log.at[i, "match_idx"])
            key = (int(pid), midx)
            prior = p_log.iloc[:i]  # All rows BEFORE this match — no look-ahead
            lookup[key] = {}
            for w in _PLAYER_ROLLING_WINDOWS:
                pw = prior.tail(w)
                for short_name in short_cols:
                    vals = pw[short_name].dropna() if short_name in pw.columns else pd.Series([], dtype=float)
                    lookup[key][f"{short_name}_{w}"] = (
                        float(vals.mean()) if len(vals) > 0 else np.nan
                    )

    print(f"  Built rolling lookup for {len(lookup)} (player, match) entries")

    # ── Build team lineup history for disruption features ─────────────────────
    # team_lineup[(team, match_idx)] = {jersey: player_id}  (starters 1-13 only)
    team_lineup: dict[tuple, dict] = {}

    starters_df = ps_joined[ps_joined["is_starter"]].copy()
    for (midx, team), grp in starters_df.groupby(["match_idx", "team"], sort=False):
        lineup = {
            int(row["jersey_number"]): int(row["player_id"])
            for _, row in grp.iterrows()
            if row["jersey_number"] > 0
        }
        team_lineup[(team, int(midx))] = lineup

    # Per-team chronologically sorted match list (for finding previous game)
    team_game_order: dict[str, list] = {}
    for (midx, team), _ in starters_df.groupby(["match_idx", "team"], sort=False):
        team_game_order.setdefault(team, []).append(int(midx))
    # Sort each team's games by match_idx (which is chronological order)
    for team in team_game_order:
        team_game_order[team].sort()

    # ── Attach features to DataFrame ──────────────────────────────────────────
    # Pre-build per-match player lists for efficient access
    # match_players[(match_idx, team)] = [(player_id, jersey, is_spine, is_starter)]
    match_players: dict[tuple, list] = {}
    for _, row in ps_joined.iterrows():
        key = (int(row["match_idx"]), row["team"])
        match_players.setdefault(key, []).append((
            int(row["player_id"]),
            int(row["jersey_number"]),
            bool(row["is_spine"]),
            bool(row["is_starter"]),
        ))

    # Determine which spine / squad stats are available
    spine_short_stats = [s for s in _SPINE_STAT_MAP.values() if s in short_cols]
    squad_short_stats = [s for s in _SQUAD_STAT_MAP.values() if s in short_cols]

    # Initialise output columns with NaN
    feature_init_cols = []
    for side in ("home", "away"):
        for stat in spine_short_stats:
            for w in _PLAYER_ROLLING_WINDOWS:
                feature_init_cols.append(f"{side}_spine_{stat}_{w}")
        for stat in squad_short_stats:
            for w in _PLAYER_ROLLING_WINDOWS:
                feature_init_cols.append(f"{side}_squad_{stat}_{w}")
        feature_init_cols.append(f"{side}_spine_changes")
        feature_init_cols.append(f"{side}_squad_turnover")

    for col in feature_init_cols:
        df[col] = np.nan

    # ── Row-by-row feature computation ────────────────────────────────────────
    for i in range(len(df)):
        match_idx = i  # df._match_idx == range(len(df))

        for side, team_col in (("home", "home_team"), ("away", "away_team")):
            team = df.at[i, team_col]
            players = match_players.get((match_idx, team), [])
            if not players:
                continue

            # ── Spine rolling form ────────────────────────────────────────────
            spine_pids = [
                pid for pid, jersey, is_sp, is_st in players
                if is_sp and is_st
            ]
            for stat in spine_short_stats:
                for w in _PLAYER_ROLLING_WINDOWS:
                    vals = [
                        lookup.get((pid, match_idx), {}).get(f"{stat}_{w}", np.nan)
                        for pid in spine_pids
                    ]
                    valid = [v for v in vals if not np.isnan(v)]
                    df.at[i, f"{side}_spine_{stat}_{w}"] = (
                        float(np.mean(valid)) if valid else np.nan
                    )

            # ── Squad (starting 13) rolling form ─────────────────────────────
            squad_pids = [
                pid for pid, jersey, is_sp, is_st in players
                if is_st
            ]
            for stat in squad_short_stats:
                for w in _PLAYER_ROLLING_WINDOWS:
                    vals = [
                        lookup.get((pid, match_idx), {}).get(f"{stat}_{w}", np.nan)
                        for pid in squad_pids
                    ]
                    valid = [v for v in vals if not np.isnan(v)]
                    df.at[i, f"{side}_squad_{stat}_{w}"] = (
                        float(np.mean(valid)) if valid else np.nan
                    )

            # ── Disruption: compare lineup with previous game ─────────────────
            game_order = team_game_order.get(team, [])
            current_pos = game_order.index(match_idx) if match_idx in game_order else -1
            if current_pos > 0:
                prev_midx = game_order[current_pos - 1]
                current_lineup = team_lineup.get((team, match_idx), {})
                prev_lineup = team_lineup.get((team, prev_midx), {})

                if current_lineup and prev_lineup:
                    # Spine changes: positions {1,6,7,9} where player differs
                    spine_changes = sum(
                        1 for jersey in _SPINE_JERSEYS
                        if current_lineup.get(jersey) != prev_lineup.get(jersey)
                    )
                    df.at[i, f"{side}_spine_changes"] = float(spine_changes)

                    # Squad turnover: starters (jerseys 1-13) with different player
                    squad_turnover = sum(
                        1 for jersey in range(1, 14)
                        if current_lineup.get(jersey) != prev_lineup.get(jersey)
                    )
                    df.at[i, f"{side}_squad_turnover"] = float(squad_turnover)

    # ── Differentials (home − away) ───────────────────────────────────────────
    for stat in spine_short_stats:
        for w in _PLAYER_ROLLING_WINDOWS:
            h = f"home_spine_{stat}_{w}"
            a = f"away_spine_{stat}_{w}"
            df[f"spine_diff_{stat}_{w}"] = df[h] - df[a]

    for stat in squad_short_stats:
        for w in _PLAYER_ROLLING_WINDOWS:
            h = f"home_squad_{stat}_{w}"
            a = f"away_squad_{stat}_{w}"
            df[f"squad_diff_{stat}_{w}"] = df[h] - df[a]

    df["spine_changes_diff"] = df["home_spine_changes"] - df["away_spine_changes"]
    df["squad_turnover_diff"] = df["home_squad_turnover"] - df["away_squad_turnover"]

    # Clean up temporary columns
    df = df.drop(columns=["_match_idx", "_round_str"], errors="ignore")

    # Coverage summary
    form_feat_sample = f"home_spine_{spine_short_stats[0]}_3" if spine_short_stats else None
    if form_feat_sample and form_feat_sample in df.columns:
        coverage = df[form_feat_sample].notna().mean() * 100
    else:
        coverage = 0.0
    n_added = (
        len(spine_short_stats) * len(_PLAYER_ROLLING_WINDOWS) * 3  # home + away + diff
        + len(squad_short_stats) * len(_PLAYER_ROLLING_WINDOWS) * 3
        + 6  # disruption: 2×home + 2×away + 2×diff
    )
    print(f"  Added {n_added} player form features | Coverage: {coverage:.0f}%")
    return df


# =========================================================================
# BUILD V4 FEATURE MATRIX
# =========================================================================

def build_v4_feature_matrix(df):
    """Build the V4 feature matrix with all V3 + V4 features."""
    print("\n" + "=" * 80)
    print("  BUILDING V4 FEATURE MATRIX")
    print("=" * 80)

    df["home_win"] = np.where(
        df["home_score"] > df["away_score"], 1.0,
        np.where(df["home_score"] < df["away_score"], 0.0, np.nan)
    )

    # === V3 FEATURES (baseline) ===
    feature_cols = []

    # Elo
    feature_cols += ["home_elo", "away_elo", "home_elo_prob", "elo_diff"]

    # Rolling form (3, 5, 8)
    for w in [3, 5, 8]:
        for stat in ("win_rate", "avg_pf", "avg_pa", "avg_margin"):
            feature_cols += [f"home_{stat}_{w}", f"away_{stat}_{w}"]
        feature_cols += [f"win_rate_diff_{w}", f"avg_margin_diff_{w}"]

    # Ladder
    feature_cols += [
        "home_ladder_pos", "away_ladder_pos", "ladder_pos_diff",
        "home_wins", "away_wins", "home_losses", "away_losses",
        "home_points_diff_season", "away_points_diff_season",
        "home_competition_points", "away_competition_points",
    ]

    # Ladder home/away splits
    feature_cols += [
        "home_team_home_win_pct", "away_team_away_win_pct",
        "home_team_home_ppg", "away_team_away_ppg",
        "home_home_win_pct", "away_home_win_pct",
        "home_away_win_pct", "away_away_win_pct",
        "home_home_ppg", "away_home_ppg", "home_away_ppg", "away_away_ppg",
        "home_home_pag", "away_home_pag", "home_away_pag", "away_away_pag",
    ]

    # Schedule
    feature_cols += [
        "home_days_rest", "away_days_rest", "rest_diff",
        "home_is_back_to_back", "away_is_back_to_back",
        "home_bye_last_round", "away_bye_last_round",
    ]

    # Context
    feature_cols += ["is_home", "round_number", "is_finals", "day_of_week", "month"]

    # V3 Odds
    feature_cols += [
        "odds_home_prob", "odds_away_prob", "odds_home_favourite",
        "odds_home_open_prob", "odds_away_open_prob",
        "spread_home_open", "total_line_open",
        "odds_home_range", "odds_away_range",
        "bookmakers_surveyed",
        "odds_movement", "odds_movement_abs",
    ]

    # H2H
    feature_cols += [
        "h2h_home_win_rate_3", "h2h_home_win_rate_5", "h2h_home_win_rate_all",
        "h2h_avg_margin_3", "h2h_avg_margin_5", "h2h_avg_margin_all",
        "h2h_matches_3", "h2h_matches_5", "h2h_matches_all",
    ]

    # Venue
    feature_cols += [
        "home_venue_win_rate", "away_venue_win_rate",
        "venue_avg_total_score", "is_neutral_venue",
    ]

    # Momentum/Trend
    feature_cols += [
        "home_form_momentum", "away_form_momentum", "form_momentum_diff",
        "home_form_momentum_3v5", "away_form_momentum_3v5",
        "home_streak", "away_streak", "streak_diff",
        "home_last_result", "away_last_result",
    ]

    # Halftime/Penalty
    feature_cols += [
        "home_avg_halftime_lead_5", "away_avg_halftime_lead_5",
        "home_avg_penalty_diff_5", "away_avg_penalty_diff_5",
        "halftime_lead_diff", "penalty_diff_diff",
    ]

    # V3 Engineered
    feature_cols += [
        "elo_diff_sq", "elo_diff_abs",
        "odds_elo_diff", "odds_elo_abs_diff",
        "home_attack_defense_3", "away_attack_defense_3", "attack_defense_diff_3",
        "season_progress", "elo_diff_x_progress",
        "comp_points_ratio", "home_strength", "away_strength", "strength_diff",
        "elo_x_rest", "ladder_x_finals",
        "home_away_split_diff", "venue_wr_diff",
    ]

    # === V4 NEW FEATURES ===

    # V4 Enhanced Odds
    feature_cols += [
        "spread_home_close", "spread_movement", "spread_movement_abs",
        "odds_spread_agree", "odds_spread_disagree_mag",
        "total_line_close", "total_movement", "total_movement_abs",
        "implied_draw_prob", "draw_competitiveness",
        "odds_home_range_close", "odds_away_range_close",
        "market_confidence",
    ]

    # V4 Scoring Consistency & Trends
    for w in [5, 8]:
        for side in ["home", "away"]:
            feature_cols += [
                f"{side}_pf_std_{w}", f"{side}_pa_std_{w}",
                f"{side}_close_game_rate_{w}",
                f"{side}_pf_trend_{w}", f"{side}_pa_trend_{w}",
            ]
        feature_cols += [
            f"pf_std_diff_{w}", f"close_game_diff_{w}",
            f"pf_trend_diff_{w}", f"pa_trend_diff_{w}",
        ]
    feature_cols += ["home_scoring_cv_5", "away_scoring_cv_5", "scoring_cv_diff_5"]

    # V4 Attendance
    feature_cols += ["attendance_normalized", "attendance_high"]

    # V4 Kickoff
    feature_cols += ["is_night_game", "is_afternoon_game", "is_day_game"]

    # V4 Lineup Stability
    feature_cols += ["home_lineup_stability", "away_lineup_stability", "lineup_stability_diff"]

    # V4 Player Impact — REMOVED from training features.
    # These scores are computed from all-time data, creating look-ahead
    # bias when used in historical training rows.  Player impact is still
    # applied via --check-lineups post-prediction adjustment.
    # feature_cols += [
    #     "home_spine_impact", "away_spine_impact", "spine_impact_diff",
    #     "home_total_impact", "away_total_impact", "total_impact_diff",
    # ]

    # V4 Engineered Interactions
    feature_cols += [
        "is_early_season", "is_mid_season", "is_late_season",
        "elo_diff_x_late", "elo_diff_x_early", "form_x_late",
        "home_defense_improving", "away_defense_improving", "defense_trend_diff",
        "scoring_env_ratio", "fav_consistency",
        "elo_spread_agree", "strong_team_rested", "home_ground_x_form",
    ]

    # V4 Team Season Stats (prior-season averages)
    for stat in _TS_COLS:
        feature_cols += [f"home_ts_{stat}", f"away_ts_{stat}", f"ts_diff_{stat}"]

    # V4 SuperCoach matchup features (12)
    for grp in ["spine", "forward", "back", "total"]:
        feature_cols += [f"home_opp_sc_{grp}", f"away_opp_sc_{grp}", f"sc_{grp}_diff"]

    # V4 Referee features
    feature_cols += ["ref_home_win_rate", "ref_games", "ref_is_high_home"]

    # V4 Player form features (42)
    # 4 spine stats × 2 windows × 3 (home/away/diff) = 24
    # 2 squad stats × 2 windows × 3 = 12
    # disruption: 2×home + 2×away + 2×diff = 6
    _spine_stats = ["run_metres", "line_breaks", "tackle_breaks", "try_assists"]
    _squad_stats = ["fantasy", "minutes"]
    for stat in _spine_stats:
        for w in [3, 5]:
            feature_cols += [f"home_spine_{stat}_{w}", f"away_spine_{stat}_{w}",
                             f"spine_diff_{stat}_{w}"]
    for stat in _squad_stats:
        for w in [3, 5]:
            feature_cols += [f"home_squad_{stat}_{w}", f"away_squad_{stat}_{w}",
                             f"squad_diff_{stat}_{w}"]
    feature_cols += [
        "home_spine_changes", "away_spine_changes", "spine_changes_diff",
        "home_squad_turnover", "away_squad_turnover", "squad_turnover_diff",
    ]

    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    # De-duplicate
    seen = set()
    unique_cols = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    feature_cols = unique_cols

    meta_cols = ["date", "season", "year", "round", "home_team", "away_team", "venue",
                 "home_score", "away_score"]
    meta_cols = [c for c in meta_cols if c in df.columns]

    all_cols = list(dict.fromkeys(meta_cols + feature_cols + ["home_win"]))
    features = df[all_cols].copy()

    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    n_before = len(features)
    features = features.dropna(subset=["home_win"]).reset_index(drop=True)
    n_dropped = n_before - len(features)

    print(f"  Dropped {n_dropped} draws")
    print(f"  Feature matrix: {features.shape[0]} rows x {features.shape[1]} cols")
    print(f"  Feature columns: {len(feature_cols)} (V3: ~131, V4 new: ~{len(feature_cols) - 131})")

    # Missing summary
    missing = features[feature_cols].isnull().sum()
    missing_pct = missing / len(features) * 100
    high_miss = missing_pct[missing_pct > 30].sort_values(ascending=False)
    if len(high_miss) > 0:
        print(f"\n  Features with >30% missing:")
        for c, pct in high_miss.items():
            print(f"    {c:<45} {pct:.1f}%")

    return features, feature_cols


# =========================================================================
# WALK-FORWARD BACKTESTING (V4)
# =========================================================================

def _sanitize_round1_features(X):
    """Clamp out-of-distribution features for season openers."""
    X = X.copy()
    MAX_REST = 14
    for col in ["home_days_rest", "away_days_rest"]:
        if col in X.columns:
            X[col] = X[col].clip(upper=MAX_REST)
    if "rest_diff" in X.columns and "home_days_rest" in X.columns and "away_days_rest" in X.columns:
        X["rest_diff"] = X["home_days_rest"] - X["away_days_rest"]
    if "round_number" in X.columns:
        is_r1 = X["round_number"] == 1
        for col in ["home_bye_last_round", "away_bye_last_round"]:
            if col in X.columns:
                X.loc[is_r1, col] = 0
        for col in ["home_lineup_stability", "away_lineup_stability",
                     "lineup_stability_diff"]:
            if col in X.columns:
                X.loc[is_r1, col] = np.nan
    return X


def _fill_odds_coherent(X):
    """Fill NaN odds features from actual closing odds (not training medians)."""
    X = X.copy()
    hp = X.get("odds_home_prob")
    ap = X.get("odds_away_prob")
    if hp is None or ap is None:
        return X
    for col, src in [("odds_home_open_prob", hp), ("odds_away_open_prob", ap)]:
        if col in X.columns:
            X[col] = X[col].fillna(src)
    if "spread_home_open" in X.columns:
        X["spread_home_open"] = X["spread_home_open"].fillna(-((hp - 0.5) * 26))
    if "spread_home_close" in X.columns:
        X["spread_home_close"] = X["spread_home_close"].fillna(-((hp - 0.5) * 26))
    for col in ["odds_movement", "odds_movement_abs",
                "spread_movement", "spread_movement_abs",
                "total_movement", "total_movement_abs"]:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    for col in ["odds_home_range", "odds_away_range",
                "odds_home_range_close", "odds_away_range_close"]:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    if "bookmakers_surveyed" in X.columns:
        X["bookmakers_surveyed"] = X["bookmakers_surveyed"].fillna(1)
    for col in ["implied_draw_prob", "draw_competitiveness"]:
        if col in X.columns:
            X[col] = X[col].fillna(0.05)
    return X


def fill_missing(X_train, X_test):
    """Fill NaN with sanitization, coherent odds, then train medians."""
    bool_cols = {"home_is_back_to_back", "away_is_back_to_back",
                 "home_bye_last_round", "away_bye_last_round",
                 "is_finals", "odds_home_favourite", "is_home", "is_neutral_venue",
                 "odds_spread_agree", "elo_spread_agree",
                 "attendance_high", "is_night_game", "is_afternoon_game", "is_day_game",
                 "is_early_season", "is_mid_season", "is_late_season"}
    medians = X_train.median()
    Xtr = _sanitize_round1_features(X_train.copy())
    Xtr = _fill_odds_coherent(Xtr)
    Xte = _sanitize_round1_features(X_test.copy())
    Xte = _fill_odds_coherent(Xte)
    for col in X_train.columns:
        fill_val = 0 if col in bool_cols else medians.get(col, 0)
        Xtr[col] = Xtr[col].fillna(fill_val)
        Xte[col] = Xte[col].fillna(fill_val)
    return Xtr, Xte


def compute_sample_weights(years, decay=0.9):
    max_yr = years.max()
    return decay ** (max_yr - years)


def select_top_features(X_train, y_train, feature_cols, sw, top_n=50):
    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.02,
                               verbosity=0, random_state=42)
    model.fit(X_train, y_train, sample_weight=sw)
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return list(imp.head(top_n).index)


def walk_forward_backtest_v4(features, feature_cols):
    """Run V4 walk-forward backtesting with expanded model set."""
    print("\n" + "=" * 80)
    print("  V4 WALK-FORWARD BACKTESTING")
    print("=" * 80)

    df = features.copy()
    all_results = {}

    # V4 Model names (V3 + new models)
    model_names = [
        "XGBoost", "LightGBM", "CatBoost", "LogReg",
        "XGB_top50", "LGB_top50", "CAT_top50",
        "RandomForest", "ExtraTrees",
        "Odds Implied",
    ]
    model_oof = {n: [] for n in model_names}
    y_parts = []
    odds_parts = []
    year_parts = []

    for fold_idx, fold in enumerate(FOLDS):
        train_end = fold["train_end"]
        test_year = fold["test_year"]

        train_mask = df["year"] <= train_end
        test_mask = df["year"] == test_year

        if test_mask.sum() == 0:
            print(f"  Fold {fold_idx+1}: No test data for {test_year}, skipping")
            y_parts.append(np.array([]))
            odds_parts.append(np.array([]))
            year_parts.append(np.array([]))
            for n in model_names:
                model_oof[n].append(np.array([]))
            continue

        X_train_raw = df.loc[train_mask, feature_cols].copy()
        y_train = df.loc[train_mask, "home_win"].values
        X_test_raw = df.loc[test_mask, feature_cols].copy()
        y_test = df.loc[test_mask, "home_win"].values

        train_years = df.loc[train_mask, "year"].values
        odds_test = df.loc[test_mask, "odds_home_prob"].values.copy()
        odds_test = np.where(np.isnan(odds_test), 0.55, odds_test)

        X_train, X_test = fill_missing(X_train_raw, X_test_raw)

        sw = compute_sample_weights(pd.Series(train_years), decay=SAMPLE_WEIGHT_DECAY)

        y_parts.append(y_test)
        odds_parts.append(odds_test)
        year_parts.append(np.full(len(y_test), test_year))

        print(f"\n  Fold {fold_idx+1}: Train <=2013-{train_end} ({len(X_train)}) -> Test {test_year} ({len(X_test)})")

        # Feature selection
        top50 = select_top_features(X_train, y_train, feature_cols, sw, top_n=50)
        X_train_top = X_train[top50]
        X_test_top = X_test[top50]

        # --- Odds Implied ---
        model_oof["Odds Implied"].append(np.clip(odds_test, 1e-7, 1-1e-7))

        # --- XGBoost (all features) ---
        m = xgb.XGBClassifier(**BEST_XGB_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["XGBoost"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # --- XGBoost (top 50) ---
        m = xgb.XGBClassifier(**BEST_XGB_PARAMS)
        m.fit(X_train_top, y_train, sample_weight=sw)
        model_oof["XGB_top50"].append(np.clip(m.predict_proba(X_test_top)[:, 1], 1e-7, 1-1e-7))

        # --- LightGBM (all features) ---
        m = lgbm.LGBMClassifier(**BEST_LGB_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["LightGBM"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # --- LightGBM (top 50) ---
        m = lgbm.LGBMClassifier(**BEST_LGB_PARAMS)
        m.fit(X_train_top, y_train, sample_weight=sw)
        model_oof["LGB_top50"].append(np.clip(m.predict_proba(X_test_top)[:, 1], 1e-7, 1-1e-7))

        # --- CatBoost (all features) ---
        m = CatBoostClassifier(**BEST_CAT_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["CatBoost"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # --- CatBoost (top 50) ---
        m = CatBoostClassifier(**BEST_CAT_PARAMS)
        m.fit(X_train_top, y_train, sample_weight=sw)
        model_oof["CAT_top50"].append(np.clip(m.predict_proba(X_test_top)[:, 1], 1e-7, 1-1e-7))

        # --- Logistic Regression ---
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        m = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        m.fit(X_train_sc, y_train, sample_weight=sw)
        model_oof["LogReg"].append(np.clip(m.predict_proba(X_test_sc)[:, 1], 1e-7, 1-1e-7))

        # --- Random Forest (V4 NEW - Optuna tuned) ---
        m = RandomForestClassifier(**BEST_RF_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["RandomForest"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # --- ExtraTrees (V4 NEW - same params as RF) ---
        m = ExtraTreesClassifier(**BEST_RF_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["ExtraTrees"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # Print fold results
        for n in ["XGBoost", "CatBoost", "RandomForest", "ExtraTrees", "Odds Implied"]:
            met = compute_metrics(y_test, model_oof[n][-1])
            print(f"    {n:15s}: Acc={met['accuracy']:.3f}  LL={met['log_loss']:.4f}")

    # Aggregate results
    print("\n" + "-" * 80)
    print("  AGGREGATE BASE MODEL RESULTS")
    print("-" * 80)

    for n in model_names:
        fold_metrics = []
        for i in range(len(FOLDS)):
            if len(y_parts[i]) == 0:
                continue
            fold_metrics.append(compute_metrics(y_parts[i], model_oof[n][i]))
        if fold_metrics:
            result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            all_results[n] = result
            print(f"  {n:15s}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}  "
                  f"Brier={result['brier']:.4f}  AUC={result['auc']:.4f}")

    return all_results, model_oof, y_parts, odds_parts, year_parts


# =========================================================================
# V4 BLENDING WITH CALIBRATION
# =========================================================================

def calibrate_probabilities(model_oof, y_parts, active_folds):
    """Apply walk-forward isotonic calibration to each model's OOF predictions."""
    print("\n  Calibrating probabilities (walk-forward isotonic)...")

    calibrated_oof = {}
    ml_models = [n for n in model_oof.keys() if n != "Odds Implied"]

    for name in ml_models:
        calibrated_oof[name] = [np.array([])] * len(y_parts)

        for fold_idx in range(len(y_parts)):
            if len(y_parts[fold_idx]) == 0:
                continue

            # Collect calibration training data from prior folds
            cal_X, cal_y = [], []
            for prev in range(fold_idx):
                if len(y_parts[prev]) == 0:
                    continue
                cal_X.append(model_oof[name][prev])
                cal_y.append(y_parts[prev])

            if len(cal_X) < 1:
                # No prior data: pass through uncalibrated
                calibrated_oof[name][fold_idx] = model_oof[name][fold_idx]
                continue

            cal_X = np.concatenate(cal_X)
            cal_y = np.concatenate(cal_y)

            # Fit isotonic regression
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(cal_X, cal_y)

            # Calibrate test fold
            calibrated_oof[name][fold_idx] = iso.predict(model_oof[name][fold_idx])

    # Keep odds as-is
    calibrated_oof["Odds Implied"] = model_oof["Odds Implied"]

    # Report calibration improvement
    for name in ["XGBoost", "CatBoost", "RandomForest"]:
        if name not in calibrated_oof:
            continue
        orig_lls, cal_lls = [], []
        for i in active_folds:
            if len(y_parts[i]) == 0:
                continue
            orig_lls.append(safe_log_loss(y_parts[i], model_oof[name][i]))
            cal_lls.append(safe_log_loss(y_parts[i], calibrated_oof[name][i]))
        if orig_lls:
            print(f"    {name}: LL {np.mean(orig_lls):.4f} -> {np.mean(cal_lls):.4f} "
                  f"({'improved' if np.mean(cal_lls) < np.mean(orig_lls) else 'unchanged'})")

    return calibrated_oof


def v4_blend_and_stack(all_results, model_oof, y_parts, odds_parts):
    """V4 blending with calibration, more combos, walk-forward OptBlend."""
    print("\n" + "=" * 80)
    print("  V4: ODDS-BLEND, CALIBRATION & STACKING")
    print("=" * 80)

    ml_models = ["XGBoost", "LightGBM", "CatBoost", "LogReg",
                 "XGB_top50", "LGB_top50", "CAT_top50",
                 "RandomForest", "ExtraTrees"]
    active_folds = [i for i in range(len(FOLDS)) if len(y_parts[i]) > 0]

    # === Step 1: Calibrate probabilities ===
    calibrated_oof = calibrate_probabilities(model_oof, y_parts, active_folds)

    # === Step 2: Per-model odds blend (both raw and calibrated) ===
    print("\n  Per-Model Odds Blends:")
    for use_cal, label_prefix in [(False, "Raw"), (True, "Cal")]:
        oof_source = calibrated_oof if use_cal else model_oof
        for name in ml_models:
            best_w, best_ll = None, float("inf")
            for w_int in range(5, 500, 5):
                w = w_int / 1000.0
                fold_lls = []
                for i in active_folds:
                    blended = w * oof_source[name][i] + (1 - w) * odds_parts[i]
                    fold_lls.append(safe_log_loss(y_parts[i], blended))
                avg_ll = np.mean(fold_lls)
                if avg_ll < best_ll:
                    best_ll = avg_ll
                    best_w = w

            fold_metrics = []
            for i in active_folds:
                blended = best_w * oof_source[name][i] + (1 - best_w) * odds_parts[i]
                fold_metrics.append(compute_metrics(y_parts[i], blended))

            result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            label = f"{label_prefix}-Blend {name} (w={best_w:.3f})"
            all_results[label] = result
            if name in ["XGBoost", "CatBoost", "RandomForest"]:
                print(f"    {label}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    # === Step 3: Multi-model OptBlend (Nelder-Mead) ===
    print("\n  Multi-Model OptBlend (Nelder-Mead):")

    def multi_blend_obj(weights, names_list, oof_dict, odds_p, y_p, active_f):
        w_odds = 1.0 - np.sum(weights)
        if w_odds < -0.5:
            return 10.0
        fold_lls = []
        for i in active_f:
            blended = np.zeros_like(y_p[i], dtype=float)
            for j, n in enumerate(names_list):
                blended += weights[j] * oof_dict[n][i]
            blended += w_odds * odds_p[i]
            blended = np.clip(blended, 1e-7, 1-1e-7)
            fold_lls.append(safe_log_loss(y_p[i], blended))
        return np.mean(fold_lls)

    blend_combos = [
        ("V4-3GBM+Odds", ["XGBoost", "LightGBM", "CatBoost"]),
        ("V4-All7+Odds", ["XGBoost", "LightGBM", "CatBoost", "LogReg",
                          "XGB_top50", "LGB_top50", "CAT_top50"]),
        ("V4-All9+Odds", ["XGBoost", "LightGBM", "CatBoost", "LogReg",
                          "XGB_top50", "LGB_top50", "CAT_top50",
                          "RandomForest", "ExtraTrees"]),
        ("V4-GBM+RF+Odds", ["XGBoost", "CatBoost", "RandomForest"]),
        ("V4-Top50+RF+Odds", ["XGB_top50", "LGB_top50", "CAT_top50", "RandomForest"]),
        ("V4-Best5+Odds", ["XGBoost", "CatBoost", "XGB_top50", "LGB_top50", "RandomForest"]),
    ]

    # Test both raw and calibrated
    for use_cal, cal_label in [(False, ""), (True, "-Cal")]:
        oof_source = calibrated_oof if use_cal else model_oof
        for combo_name, combo in blend_combos:
            n_m = len(combo)
            x0 = np.array([0.1 / n_m] * n_m)
            res = minimize(
                multi_blend_obj, x0,
                args=(combo, oof_source, odds_parts, y_parts, active_folds),
                method="Nelder-Mead",
                options={"maxiter": 10000, "xatol": 0.0005, "fatol": 1e-8}
            )
            bw = res.x
            w_odds = 1.0 - np.sum(bw)

            fold_metrics = []
            for i in active_folds:
                blended = np.zeros_like(y_parts[i], dtype=float)
                for j, n in enumerate(combo):
                    blended += bw[j] * oof_source[n][i]
                blended += w_odds * odds_parts[i]
                blended = np.clip(blended, 1e-7, 1-1e-7)
                fold_metrics.append(compute_metrics(y_parts[i], blended))

            result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            weight_str = ", ".join([f"{n}={bw[j]:.3f}" for j, n in enumerate(combo)])
            label = f"OptBlend{cal_label} {combo_name} ({weight_str}, odds={w_odds:.3f})"
            all_results[label] = result
            print(f"    OptBlend{cal_label} {combo_name}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    # === Step 4: Walk-Forward OptBlend (no look-ahead) ===
    print("\n  Walk-Forward OptBlend (no look-ahead in weights):")
    for combo_name, combo in [
        ("WF-All9+Odds", ["XGBoost", "LightGBM", "CatBoost", "LogReg",
                          "XGB_top50", "LGB_top50", "CAT_top50",
                          "RandomForest", "ExtraTrees"]),
        ("WF-Best5+Odds", ["XGBoost", "CatBoost", "XGB_top50", "LGB_top50", "RandomForest"]),
    ]:
        wf_probs = [np.array([])] * len(FOLDS)
        for fold_idx in range(len(FOLDS)):
            if len(y_parts[fold_idx]) == 0:
                continue

            if fold_idx < 2:
                # Not enough history: equal weight blend
                n_m = len(combo)
                equal_w = 0.15 / n_m
                blended = np.zeros_like(y_parts[fold_idx], dtype=float)
                for n in combo:
                    blended += equal_w * model_oof[n][fold_idx]
                blended += (1 - 0.15) * odds_parts[fold_idx]
            else:
                # Optimize weights on prior folds only
                prior_folds = [p for p in range(fold_idx) if len(y_parts[p]) > 0]
                n_m = len(combo)
                x0 = np.array([0.05 / n_m] * n_m)
                res = minimize(
                    multi_blend_obj, x0,
                    args=(combo, model_oof, odds_parts, y_parts, prior_folds),
                    method="Nelder-Mead",
                    options={"maxiter": 5000, "xatol": 0.001, "fatol": 1e-7}
                )
                bw = res.x
                w_odds = 1.0 - np.sum(bw)
                blended = np.zeros_like(y_parts[fold_idx], dtype=float)
                for j, n in enumerate(combo):
                    blended += bw[j] * model_oof[n][fold_idx]
                blended += w_odds * odds_parts[fold_idx]

            wf_probs[fold_idx] = np.clip(blended, 1e-7, 1-1e-7)

        fold_metrics = []
        for i in active_folds:
            fold_metrics.append(compute_metrics(y_parts[i], wf_probs[i]))
        result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        label = f"WF-OptBlend {combo_name}"
        all_results[label] = result
        print(f"    {label}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    # === Step 5: MLP Neural Network Stacking ===
    print("\n  Neural Network Stacking:")
    for stack_label, stack_models in [
        ("MLP-All9+Odds", ["XGBoost", "LightGBM", "CatBoost", "LogReg",
                           "XGB_top50", "LGB_top50", "CAT_top50",
                           "RandomForest", "ExtraTrees"]),
        ("MLP-Best5+Odds", ["XGBoost", "CatBoost", "XGB_top50", "LGB_top50", "RandomForest"]),
    ]:
        stack_probs = [np.array([])] * len(FOLDS)
        for fold_idx in range(len(FOLDS)):
            if len(y_parts[fold_idx]) == 0:
                continue

            meta_X_train_parts, meta_y_train_parts = [], []
            for prev in range(fold_idx):
                if len(y_parts[prev]) == 0:
                    continue
                row = np.column_stack([model_oof[n][prev] for n in stack_models] + [odds_parts[prev]])
                meta_X_train_parts.append(row)
                meta_y_train_parts.append(y_parts[prev])

            meta_X_test = np.column_stack(
                [model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]]
            )

            if len(meta_X_train_parts) < 2:
                # Not enough data: simple average
                avg = np.mean([model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]], axis=0)
                stack_probs[fold_idx] = np.clip(avg, 1e-7, 1-1e-7)
                continue

            meta_X_train = np.vstack(meta_X_train_parts)
            meta_y_train = np.concatenate(meta_y_train_parts)

            # Scale meta features
            meta_scaler = StandardScaler()
            meta_X_train_sc = meta_scaler.fit_transform(meta_X_train)
            meta_X_test_sc = meta_scaler.transform(meta_X_test)

            # MLP meta-learner
            mlp = MLPClassifier(
                hidden_layer_sizes=(32, 16), activation="relu",
                max_iter=500, random_state=42, early_stopping=True,
                validation_fraction=0.15, learning_rate_init=0.001,
            )
            mlp.fit(meta_X_train_sc, meta_y_train)
            stack_probs[fold_idx] = np.clip(mlp.predict_proba(meta_X_test_sc)[:, 1], 1e-7, 1-1e-7)

        fold_metrics = []
        for i in active_folds:
            if len(stack_probs[i]) == 0:
                continue
            fold_metrics.append(compute_metrics(y_parts[i], stack_probs[i]))
        if fold_metrics:
            result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            label = f"Stacking ({stack_label} -> MLP)"
            all_results[label] = result
            print(f"    {label}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    # === Step 6: LR Stacking (as in V3, for comparison) ===
    print("\n  LogReg Stacking:")
    for stack_label, stack_models in [
        ("LR-All9+Odds", ["XGBoost", "LightGBM", "CatBoost", "LogReg",
                          "XGB_top50", "LGB_top50", "CAT_top50",
                          "RandomForest", "ExtraTrees"]),
    ]:
        stack_probs = [np.array([])] * len(FOLDS)
        for fold_idx in range(len(FOLDS)):
            if len(y_parts[fold_idx]) == 0:
                continue

            meta_X_train_parts, meta_y_train_parts = [], []
            for prev in range(fold_idx):
                if len(y_parts[prev]) == 0:
                    continue
                row = np.column_stack([model_oof[n][prev] for n in stack_models] + [odds_parts[prev]])
                meta_X_train_parts.append(row)
                meta_y_train_parts.append(y_parts[prev])

            meta_X_test = np.column_stack(
                [model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]]
            )

            if len(meta_X_train_parts) < 1:
                avg = np.mean([model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]], axis=0)
                stack_probs[fold_idx] = np.clip(avg, 1e-7, 1-1e-7)
                continue

            meta_X_train = np.vstack(meta_X_train_parts)
            meta_y_train = np.concatenate(meta_y_train_parts)

            meta_lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            meta_lr.fit(meta_X_train, meta_y_train)
            stack_probs[fold_idx] = np.clip(meta_lr.predict_proba(meta_X_test)[:, 1], 1e-7, 1-1e-7)

        fold_metrics = []
        for i in active_folds:
            if len(stack_probs[i]) == 0:
                continue
            fold_metrics.append(compute_metrics(y_parts[i], stack_probs[i]))
        if fold_metrics:
            result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            label = f"Stacking ({stack_label} -> LR)"
            all_results[label] = result
            print(f"    {label}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    return all_results


# =========================================================================
# RESULTS COMPARISON
# =========================================================================

def print_comparison(all_results):
    """Print comprehensive comparison with V3 baselines."""
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE V4 RESULTS COMPARISON")
    print("=" * 80)

    rows = []
    for name, metrics in all_results.items():
        rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Log Loss": metrics["log_loss"],
            "Brier": metrics.get("brier", np.nan),
            "AUC": metrics.get("auc", np.nan),
        })

    comp_df = pd.DataFrame(rows).sort_values("Log Loss", ascending=True).reset_index(drop=True)

    odds_ll = all_results.get("Odds Implied", {}).get("log_loss", 0.5999)
    odds_acc = all_results.get("Odds Implied", {}).get("accuracy", 0.6800)

    # V3 best for reference
    v3_best_ll = 0.5977
    v3_best_acc = 0.6834

    print()
    hdr = f"{'#':>3}  {'Model':<70} | {'Acc':>7} | {'LL':>7} | {'Brier':>7} | {'AUC':>7} | {'vs Odds':>8} | {'vs V3':>8}"
    print(hdr)
    print("-" * len(hdr))

    for idx, row in comp_df.head(30).iterrows():
        ll_diff = row["Log Loss"] - odds_ll
        v3_diff = row["Log Loss"] - v3_best_ll
        marker = ""
        if row["Model"] == "Odds Implied":
            marker = " <-BASE"
        elif row["Log Loss"] < v3_best_ll:
            marker = " ***V4>"

        print(
            f"{idx+1:3d}  {row['Model']:<70} | {row['Accuracy']:7.4f} | "
            f"{row['Log Loss']:7.4f} | {row['Brier']:7.4f} | {row['AUC']:7.4f} | "
            f"{ll_diff:+8.4f} | {v3_diff:+8.4f}{marker}"
        )

    print("-" * len(hdr))

    best = comp_df.iloc[0]
    print(f"\n  BEST V4 MODEL: {best['Model']}")
    print(f"    Accuracy:  {best['Accuracy']:.4f}  (V3 best: {v3_best_acc:.4f}, odds: {odds_acc:.4f})")
    print(f"    Log Loss:  {best['Log Loss']:.4f}  (V3 best: {v3_best_ll:.4f}, odds: {odds_ll:.4f})")

    if best["Log Loss"] < odds_ll:
        imp = (odds_ll - best["Log Loss"]) / odds_ll * 100
        print(f"\n    >>> BEATS ODDS BASELINE by {imp:.3f}% in log loss <<<")

    if best["Log Loss"] < v3_best_ll:
        imp_v3 = (v3_best_ll - best["Log Loss"]) / v3_best_ll * 100
        print(f"    >>> BEATS V3 BEST by {imp_v3:.3f}% in log loss <<<")
    else:
        gap = best["Log Loss"] - v3_best_ll
        print(f"\n    V3 still leads by {gap:.4f} in log loss")

    beats_odds = comp_df[comp_df["Log Loss"] < odds_ll]
    beats_v3 = comp_df[comp_df["Log Loss"] < v3_best_ll]
    print(f"\n    {len(beats_odds)} models beat odds baseline, {len(beats_v3)} beat V3 best")

    return comp_df


# =========================================================================
# MAIN PIPELINE
# =========================================================================

def main():
    overall_start = time.time()

    print()
    print("*" * 80)
    print("*  NRL MATCH PREDICTION - V4 ENHANCED PIPELINE")
    print("*  Goal: Beat V3 best (68.3% acc / 0.5977 LL) and odds baseline (68.0% / 0.5999)")
    print("*" * 80)
    print()

    # === STEP 1: Load data (reuse V3 function) ===
    matches, ladders, odds = v3.load_and_fix_homeaway()

    # === STEP 2: Link odds ===
    matches = v3.link_odds(matches, odds)

    # === STEP 3: Tune Elo (reuse V3) ===
    elo_params = v3.tune_elo(matches, n_trials=50)

    # === STEP 4: Backfill Elo ===
    matches = v3.backfill_elo(matches, elo_params)

    # === STEP 5: V3 base features ===
    matches = v3.compute_rolling_form_features(matches)
    matches = v3.compute_h2h_features(matches)
    matches = v3.compute_ladder_features(matches, ladders)
    matches = v3.compute_venue_features(matches)
    matches = v3.compute_odds_features(matches)
    matches = v3.compute_schedule_features(matches)
    matches = v3.compute_contextual_features(matches)
    matches = v3.compute_engineered_features(matches)

    # === STEP 6: V4 NEW features ===
    matches = compute_v4_odds_features(matches)
    matches = compute_scoring_consistency_features(matches)
    matches = compute_attendance_features(matches)
    matches = compute_kickoff_features(matches)
    matches = compute_lineup_stability_features(matches)
    matches = compute_player_impact_features(matches)
    matches = compute_v4_engineered_features(matches)
    matches = compute_sc_matchup_features(matches)

    # === STEP 7: Build V4 feature matrix ===
    features, feature_cols = build_v4_feature_matrix(matches)

    # Save V4 features
    output_path = FEATURES_DIR / "features_v4.parquet"
    features.to_parquet(output_path, index=False)
    print(f"\n  Saved features_v4.parquet: {output_path}")
    print(f"  Shape: {features.shape}")

    # === STEP 8: Walk-forward backtesting ===
    all_results, model_oof, y_parts, odds_parts, year_parts = walk_forward_backtest_v4(
        features, feature_cols
    )

    # === STEP 9: Blending, calibration, stacking ===
    all_results = v4_blend_and_stack(all_results, model_oof, y_parts, odds_parts)

    # === STEP 10: Print comparison ===
    comp_df = print_comparison(all_results)

    # Save results
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(report_dir / "v4_results_comparison.csv", index=False)

    elapsed = time.time() - overall_start
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\n" + "=" * 80)
    print("  V4 PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
