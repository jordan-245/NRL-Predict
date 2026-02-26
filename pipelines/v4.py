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
V4_BLEND_WEIGHTS = {
    "CAT_top50":  0.495,
}
V4_BLEND_ODDS_WEIGHT = 0.505  # 1 - sum(model weights)


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
