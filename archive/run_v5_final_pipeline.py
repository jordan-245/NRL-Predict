"""
NRL Match Prediction - V5 FINAL Pipeline
==========================================
The most comprehensive prediction pipeline, integrating ALL available data
sources: player quality, weather, referee, travel, match stats, plus
odds-decorrelated model variants and advanced stacking.

Builds on V3 (131 features) + V4 (194 features) + V5 domain features,
adding player quality, weather, referee, travel distance, enhanced match
stats, odds-free model variants, and LightGBM stacking meta-learner.

Target: Beat V4 best (LL=0.5960) and approach 70% accuracy.

Usage:
    python run_v5_final_pipeline.py
"""

import sys
import time
import warnings
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from scipy.optimize import minimize
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING)

# Import V3 and V4 base functions
import run_enhance_and_retrain as v3
import run_v4_pipeline as v4

FOLDS = [
    {"train_end": 2017, "test_year": 2018},
    {"train_end": 2018, "test_year": 2019},
    {"train_end": 2019, "test_year": 2020},
    {"train_end": 2020, "test_year": 2021},
    {"train_end": 2021, "test_year": 2022},
    {"train_end": 2022, "test_year": 2023},
    {"train_end": 2023, "test_year": 2024},
    {"train_end": 2024, "test_year": 2025},
]

# Hybrid hyperparameters: V4 params for GBMs (best in blend), V5-tuned for RF
BEST_XGB_PARAMS = v4.BEST_XGB_PARAMS.copy()
BEST_LGB_PARAMS = v4.BEST_LGB_PARAMS.copy()
BEST_CAT_PARAMS = v4.BEST_CAT_PARAMS.copy()
# V5-tuned RF params (improved from 0.6066 to 0.6053 log loss)
BEST_RF_PARAMS = {
    'n_estimators': 234, 'max_depth': 10, 'min_samples_leaf': 23,
    'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1,
}
SAMPLE_WEIGHT_DECAY = 0.920  # V5-tuned (was 0.907)

# V5-tuned GBM params (different error profile for diversity)
V5_XGB_PARAMS = {
    'n_estimators': 628, 'max_depth': 2, 'learning_rate': 0.0212,
    'subsample': 0.581, 'colsample_bytree': 0.474,
    'reg_alpha': 0.216, 'reg_lambda': 0.082,
    'min_child_weight': 37, 'gamma': 6.249,
    'eval_metric': 'logloss', 'verbosity': 0, 'random_state': 42,
}
V5_CAT_PARAMS = {
    'iterations': 387, 'depth': 7, 'learning_rate': 0.00904,
    'l2_leaf_reg': 1.923, 'subsample': 0.475,
    'colsample_bylevel': 0.413, 'min_data_in_leaf': 15,
    'random_seed': 42, 'verbose': 0, 'allow_writing_files': False,
}

# Odds-related column substrings. Any feature column containing one of these
# is considered odds-derived and will be EXCLUDED from the odds-free model set.
ODDS_FEATURE_SUBSTRINGS = [
    "odds", "spread", "total_line", "h2h_", "implied_draw",
    "draw_competitiveness", "market_confidence", "bookmakers",
    "overround", "odds_movement", "fav_consistency", "elo_spread_agree",
    "odds_spread", "scoring_env_ratio",
]


# =========================================================================
# TEAM STATE MAPPING (for interstate travel features)
# =========================================================================

TEAM_STATE = {
    "Brisbane Broncos": "QLD",
    "North Queensland Cowboys": "QLD",
    "Gold Coast Titans": "QLD",
    "Dolphins": "QLD",
    "Melbourne Storm": "VIC",
    "New Zealand Warriors": "NZ",
    "Penrith Panthers": "NSW",
    "Sydney Roosters": "NSW",
    "South Sydney Rabbitohs": "NSW",
    "Canterbury Bulldogs": "NSW",
    "Manly Sea Eagles": "NSW",
    "Newcastle Knights": "NSW",
    "Canberra Raiders": "ACT",
    "Wests Tigers": "NSW",
    "Cronulla Sharks": "NSW",
    "Parramatta Eels": "NSW",
    "St George Illawarra Dragons": "NSW",
}

STATE_DISTANCE = {
    ("NSW", "NSW"): 0, ("NSW", "ACT"): 280, ("NSW", "QLD"): 900,
    ("NSW", "VIC"): 880, ("NSW", "NZ"): 2150,
    ("ACT", "ACT"): 0, ("ACT", "QLD"): 1200, ("ACT", "VIC"): 660,
    ("ACT", "NZ"): 2400,
    ("QLD", "QLD"): 0, ("QLD", "VIC"): 1700, ("QLD", "NZ"): 2700,
    ("VIC", "VIC"): 0, ("VIC", "NZ"): 2600,
    ("NZ", "NZ"): 0,
}


def get_travel_distance(state1, state2):
    key = tuple(sorted([state1, state2]))
    return STATE_DISTANCE.get(key, STATE_DISTANCE.get((key[1], key[0]), 1000))


# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================

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


def _try_load_parquet(path, label):
    """Attempt to load a parquet file, return empty DataFrame on failure."""
    p = Path(path)
    if p.exists():
        try:
            df = pd.read_parquet(p)
            print(f"  Loaded {label}: {df.shape[0]} rows x {df.shape[1]} cols")
            return df
        except Exception as e:
            print(f"  WARNING: Failed to load {label} from {p}: {e}")
    else:
        print(f"  WARNING: {label} not found at {p}, skipping")
    return pd.DataFrame()


def _build_slug_to_canonical():
    """Build a mapping from team slug fragments to canonical team names."""
    try:
        from config.team_mappings import TEAM_SLUGS, TEAM_ALIASES
    except ImportError:
        return {}
    mapping = {}
    for canonical, slug in TEAM_SLUGS.items():
        mapping[slug] = canonical
        # Add slug fragments (e.g. "panthers" -> "Penrith Panthers")
        for part in slug.split("-"):
            if len(part) > 3:
                mapping[part] = canonical
    # Also add lowercase aliases
    for canonical, aliases in TEAM_ALIASES.items():
        for alias in aliases:
            mapping[alias.lower()] = canonical
        mapping[canonical.lower()] = canonical
    return mapping


SLUG_TO_CANONICAL = _build_slug_to_canonical()


def _resolve_team_name(name):
    """Resolve a team name (slug, alias, or canonical) to canonical form."""
    if pd.isna(name):
        return name
    name_str = str(name).strip()
    # Try exact canonical match
    if name_str in SLUG_TO_CANONICAL.values():
        return name_str
    # Try lookup
    key = name_str.lower().replace(" ", "-")
    if key in SLUG_TO_CANONICAL:
        return SLUG_TO_CANONICAL[key]
    key = name_str.lower()
    if key in SLUG_TO_CANONICAL:
        return SLUG_TO_CANONICAL[key]
    # Try partial match on slug fragments
    for slug_key, canonical in SLUG_TO_CANONICAL.items():
        if slug_key in key or key in slug_key:
            return canonical
    return name_str


def _standardise_merge_keys(df, year_col="year", round_col="round_slug",
                            home_col="home_slug", away_col="away_slug"):
    """Standardise merge key columns for joining external data."""
    out = df.copy()
    if year_col in out.columns:
        out["_merge_year"] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
    if round_col in out.columns:
        out["_merge_round"] = out[round_col].astype(str).str.strip()
    if home_col in out.columns:
        out["_merge_home"] = out[home_col].apply(_resolve_team_name)
    if away_col in out.columns:
        out["_merge_away"] = out[away_col].apply(_resolve_team_name)
    return out


# =========================================================================
# V5-FINAL FEATURE: PLAYER QUALITY
# =========================================================================

def compute_player_quality_features(matches):
    """Merge or compute player quality features from parquet files."""
    print("\n" + "=" * 80)
    print("  V5-FINAL: COMPUTING PLAYER QUALITY FEATURES")
    print("=" * 80)

    df = matches.copy()

    # Try pre-computed player quality features first
    pq_path = PROCESSED_DIR / "player_quality_features.parquet"
    pq_df = _try_load_parquet(pq_path, "player_quality_features")

    if pq_df.empty:
        # Try computing from player_match_stats
        ps_path = PROCESSED_DIR / "player_match_stats.parquet"
        ps_df = _try_load_parquet(ps_path, "player_match_stats")

        if ps_df.empty:
            print("  No player data available, skipping player quality features")
            return df

        pq_df = _compute_player_quality_from_stats(ps_df, df)

    if pq_df.empty:
        print("  Could not compute player quality features")
        return df

    # Merge player quality features onto matches
    df = _merge_external_data(df, pq_df, "player_quality")
    return df


def _compute_player_quality_from_stats(player_stats, matches):
    """Compute rolling player quality features from per-player stats."""
    ps = player_stats.copy()

    # Standardise team names in player stats
    for col in ["home_slug", "away_slug"]:
        if col in ps.columns:
            ps[col] = ps[col].apply(_resolve_team_name)

    # Determine which columns identify the team side for each player
    if "side" not in ps.columns:
        print("  Player stats missing 'side' column, cannot compute quality")
        return pd.DataFrame()

    # Spine positions
    spine_positions = {"Fullback", "Halfback", "Hooker", "Five-Eighth",
                       "fullback", "halfback", "hooker", "five-eighth",
                       "1", "7", "9", "6"}
    forward_positions = {"Prop", "Second Row", "Lock",
                         "prop", "second row", "lock",
                         "8", "10", "11", "12", "13"}

    fantasy_col = None
    for c in ["fantasyPointsTotal", "fantasy_points_total", "fantasy_points"]:
        if c in ps.columns:
            fantasy_col = c
            break

    run_metres_col = None
    for c in ["allRunMetres", "all_run_metres", "runMetres", "run_metres"]:
        if c in ps.columns:
            run_metres_col = c
            break

    if fantasy_col is None:
        print("  No fantasy points column found in player stats")
        return pd.DataFrame()

    ps[fantasy_col] = pd.to_numeric(ps[fantasy_col], errors="coerce")
    if run_metres_col:
        ps[run_metres_col] = pd.to_numeric(ps[run_metres_col], errors="coerce")

    # Build per-match team-level aggregates
    # Determine match key columns
    year_col = "year" if "year" in ps.columns else None
    round_col = next((c for c in ["round_slug", "round"] if c in ps.columns), None)
    home_col = next((c for c in ["home_slug", "home_team"] if c in ps.columns), None)
    away_col = next((c for c in ["away_slug", "away_team"] if c in ps.columns), None)

    if not all([year_col, round_col, home_col, away_col]):
        print("  Missing match key columns in player stats")
        return pd.DataFrame()

    # Group by match and side
    group_keys = [year_col, round_col, home_col, away_col, "side"]
    available_keys = [k for k in group_keys if k in ps.columns]

    agg_dict = {fantasy_col: "sum"}
    if run_metres_col:
        agg_dict[run_metres_col] = "sum"

    match_agg = ps.groupby(available_keys).agg(agg_dict).reset_index()

    # Also compute spine-specific aggregates
    is_spine = ps["position"].astype(str).apply(
        lambda p: any(sp in str(p) for sp in spine_positions)
    )
    spine_stats = ps[is_spine].groupby(available_keys).agg(
        {fantasy_col: "sum"}
    ).reset_index()
    spine_stats = spine_stats.rename(columns={fantasy_col: "spine_fantasy"})

    is_forward = ps["position"].astype(str).apply(
        lambda p: any(fp in str(p) for fp in forward_positions)
    )
    if run_metres_col:
        forward_stats = ps[is_forward].groupby(available_keys).agg(
            {run_metres_col: "sum"}
        ).reset_index()
        forward_stats = forward_stats.rename(columns={run_metres_col: "forward_run_metres"})
    else:
        forward_stats = pd.DataFrame()

    # Now build rolling features per team
    # This requires walk-forward computation from the matches dataframe
    # For simplicity, compute team-level fantasy per match then rolling average
    print("  Computing rolling player quality from match-level aggregates...")

    # Build team match log with fantasy totals
    team_fantasy_log = defaultdict(list)  # team -> [(match_idx, fantasy_total, spine, fwd_metres)]
    team_spine_log = defaultdict(list)
    team_fwd_log = defaultdict(list)

    # Process home and away sides from the aggregated data
    for side_val in ["home", "away", "Home", "Away"]:
        side_data = match_agg[match_agg["side"].astype(str).str.lower() == side_val.lower()]
        team_col_for_side = home_col if side_val.lower() == "home" else away_col

        for _, row in side_data.iterrows():
            team = _resolve_team_name(row[team_col_for_side])
            yr = row[year_col]
            rnd = str(row[round_col])
            ft = row[fantasy_col]
            team_fantasy_log[(team, yr, rnd)] = ft

    for side_val in ["home", "away", "Home", "Away"]:
        side_spine = spine_stats[spine_stats["side"].astype(str).str.lower() == side_val.lower()]
        team_col_for_side = home_col if side_val.lower() == "home" else away_col
        for _, row in side_spine.iterrows():
            team = _resolve_team_name(row[team_col_for_side])
            yr = row[year_col]
            rnd = str(row[round_col])
            team_spine_log[(team, yr, rnd)] = row.get("spine_fantasy", 0)

    if not forward_stats.empty:
        for side_val in ["home", "away", "Home", "Away"]:
            side_fwd = forward_stats[forward_stats["side"].astype(str).str.lower() == side_val.lower()]
            team_col_for_side = home_col if side_val.lower() == "home" else away_col
            for _, row in side_fwd.iterrows():
                team = _resolve_team_name(row[team_col_for_side])
                yr = row[year_col]
                rnd = str(row[round_col])
                team_fwd_log[(team, yr, rnd)] = row.get("forward_run_metres", 0)

    # Now compute rolling averages per match in the matches dataframe
    team_fantasy_history = defaultdict(list)
    team_spine_history = defaultdict(list)
    team_fwd_history = defaultdict(list)

    result_rows = []

    for idx, mrow in matches.iterrows():
        home = mrow.get("home_team", "")
        away = mrow.get("away_team", "")
        yr = mrow.get("year")
        rnd = str(mrow.get("round", ""))

        row_data = {}

        # Pre-match rolling averages (window=5)
        for side, team in [("home", home), ("away", away)]:
            hist_f = team_fantasy_history[team]
            hist_s = team_spine_history[team]
            hist_fwd = team_fwd_history[team]

            if len(hist_f) >= 3:
                row_data[f"{side}_team_avg_fantasy_5"] = np.mean(hist_f[-5:])
            else:
                row_data[f"{side}_team_avg_fantasy_5"] = np.nan

            if len(hist_s) >= 3:
                row_data[f"{side}_spine_quality_5"] = np.mean(hist_s[-5:])
            else:
                row_data[f"{side}_spine_quality_5"] = np.nan

            if len(hist_fwd) >= 3:
                row_data[f"{side}_forward_run_metres_5"] = np.mean(hist_fwd[-5:])
            else:
                row_data[f"{side}_forward_run_metres_5"] = np.nan

        # Differentials
        hf = row_data.get("home_team_avg_fantasy_5")
        af = row_data.get("away_team_avg_fantasy_5")
        if pd.notna(hf) and pd.notna(af):
            row_data["team_fantasy_diff_5"] = hf - af
        else:
            row_data["team_fantasy_diff_5"] = np.nan

        hs = row_data.get("home_spine_quality_5")
        as_ = row_data.get("away_spine_quality_5")
        if pd.notna(hs) and pd.notna(as_):
            row_data["spine_quality_diff_5"] = hs - as_
        else:
            row_data["spine_quality_diff_5"] = np.nan

        result_rows.append(row_data)

        # Post-match: update histories
        for side_label, team in [("home", home), ("away", away)]:
            ft_val = team_fantasy_log.get((team, yr, rnd))
            if ft_val is not None and pd.notna(ft_val):
                team_fantasy_history[team].append(ft_val)

            sp_val = team_spine_log.get((team, yr, rnd))
            if sp_val is not None and pd.notna(sp_val):
                team_spine_history[team].append(sp_val)

            fwd_val = team_fwd_log.get((team, yr, rnd))
            if fwd_val is not None and pd.notna(fwd_val):
                team_fwd_history[team].append(fwd_val)

    result_df = pd.DataFrame(result_rows, index=matches.index)
    n_cols = len([c for c in result_df.columns if result_df[c].notna().any()])
    print(f"  Computed {n_cols} player quality features from stats")
    return result_df



def _normalize_round(r):
    """Normalize round slug to canonical format used by matches.parquet.

    'round-1' -> '1', 'round-10' -> '10'
    'finals-week-2' -> 'semi-final'
    'finals-week-3' -> 'prelim-final'
    'grand-final' -> 'grand-final' (unchanged)
    'semi-final' -> 'semi-final' (unchanged)
    '1' -> '1' (unchanged)
    """
    r = str(r).strip()
    if r.startswith("round-"):
        return r[6:]  # strip 'round-' prefix
    slug_map = {
        "finals-week-1": "qualifying-final",
        "finals-week-2": "semi-final",
        "finals-week-3": "prelim-final",
    }
    return slug_map.get(r, r)


def _merge_external_data(matches, ext_df, label):
    """Merge external data onto matches using flexible key matching.

    Handles the case where external data may have home/away teams in a
    different order than the main matches DataFrame (e.g. raw vs corrected
    home/away designations). Tries direct merge first, then swapped merge
    for unmatched rows, flipping home/away feature values accordingly.
    """
    df = matches.copy()

    if ext_df.empty:
        return df

    # Identify feature columns (not key columns)
    key_candidates = {"year", "round", "round_slug", "home_team", "away_team",
                      "home_slug", "away_slug", "date", "side", "venue",
                      "venueCity", "_merge_year", "_merge_round",
                      "_merge_home", "_merge_away", "parsed_date"}
    feature_cols = [c for c in ext_df.columns if c not in key_candidates]

    if not feature_cols:
        print(f"  No feature columns found in {label}")
        return df

    # Standardise team names in external data
    ext = ext_df.copy()
    for col in ["home_team", "home_slug"]:
        if col in ext.columns:
            ext["_ext_home"] = ext[col].apply(_resolve_team_name)
            break
    else:
        ext["_ext_home"] = np.nan

    for col in ["away_team", "away_slug"]:
        if col in ext.columns:
            ext["_ext_away"] = ext[col].apply(_resolve_team_name)
            break
    else:
        ext["_ext_away"] = np.nan

    if "year" in ext.columns:
        ext["_ext_year"] = pd.to_numeric(ext["year"], errors="coerce").astype(int)
    if "round_slug" in ext.columns:
        ext["_ext_round"] = ext["round_slug"].astype(str).str.strip().apply(_normalize_round)
    elif "round" in ext.columns:
        ext["_ext_round"] = ext["round"].astype(str).str.strip().apply(_normalize_round)

    merge_ext = ext[["_ext_year", "_ext_round", "_ext_home", "_ext_away"] + feature_cols].copy()
    merge_ext = merge_ext.drop_duplicates(subset=["_ext_year", "_ext_round", "_ext_home", "_ext_away"])

    # Prepare main df keys (normalize round format too)
    round_str = df["round"].astype(str).str.strip().apply(_normalize_round)
    year_vals = df["year"].values

    # --- Step 1: Direct merge via dict lookup (home=home, away=away) ---
    ext_lookup = {}
    for _, erow in merge_ext.iterrows():
        key = (int(erow["_ext_year"]), erow["_ext_round"],
               erow["_ext_home"], erow["_ext_away"])
        ext_lookup[key] = {fc: erow[fc] for fc in feature_cols}

    # Also build swapped lookup (flip home <-> away keys + swap feature values)
    swap_lookup = {}
    for _, erow in merge_ext.iterrows():
        key = (int(erow["_ext_year"]), erow["_ext_round"],
               erow["_ext_away"], erow["_ext_home"])  # swapped teams
        vals = {}
        home_fc = [c for c in feature_cols if c.startswith("home_")]
        away_fc = [c for c in feature_cols if c.startswith("away_")]
        diff_fc = [c for c in feature_cols
                   if "diff" in c.lower() and c not in home_fc and c not in away_fc]
        # Copy all values first
        for fc in feature_cols:
            vals[fc] = erow[fc]
        # Swap home_ <-> away_
        for h_col in home_fc:
            a_col = "away_" + h_col[5:]
            if a_col in feature_cols:
                vals[h_col] = erow[a_col]
                vals[a_col] = erow[h_col]
        # Negate diff columns
        for dc in diff_fc:
            if pd.notna(erow[dc]):
                vals[dc] = -erow[dc]
        swap_lookup[key] = vals

    # --- Step 2: Look up each match (direct first, then swapped) ---
    n_direct = 0
    n_swap = 0
    results = {fc: [np.nan] * len(df) for fc in feature_cols}

    for i, (_, row) in enumerate(df.iterrows()):
        key = (int(year_vals[i]), round_str.iloc[i],
               row["home_team"], row["away_team"])

        vals = ext_lookup.get(key)
        if vals is not None:
            for fc in feature_cols:
                results[fc][i] = vals[fc]
            n_direct += 1
            continue

        vals = swap_lookup.get(key)
        if vals is not None:
            for fc in feature_cols:
                results[fc][i] = vals[fc]
            n_swap += 1

    # Assign results
    for fc in feature_cols:
        df[fc] = results[fc]

    n_total = n_direct + n_swap
    print(f"  Merged {label}: {n_total}/{len(df)} matched "
          f"({n_direct} direct + {n_swap} swapped)")

    return df


# =========================================================================
# V5-FINAL FEATURE: WEATHER
# =========================================================================

def compute_weather_features(matches):
    """Load and merge weather features."""
    print("\n" + "=" * 80)
    print("  V5-FINAL: COMPUTING WEATHER FEATURES")
    print("=" * 80)

    df = matches.copy()

    # Load weather data
    weather_df = _try_load_parquet(PROCESSED_DIR / "weather_data.parquet", "weather_data")

    if weather_df.empty:
        # Try match_metadata for ground conditions
        meta_df = _try_load_parquet(PROCESSED_DIR / "match_metadata.parquet", "match_metadata (weather)")
        if not meta_df.empty:
            df = _merge_ground_conditions(df, meta_df)
        else:
            print("  No weather data available, skipping")
        return df

    # Rename columns to expected names
    rename_map = {
        "temperature_2m": "temperature",
        "apparent_temperature": "apparent_temp",
        "rain": "rain_mm",
        "precipitation": "precipitation_mm",
        "relative_humidity_2m": "humidity_pct",
        "wind_speed_10m": "wind_speed_kmh",
        "wind_gusts_10m": "wind_gusts_kmh",
    }
    weather_df = weather_df.rename(columns={k: v for k, v in rename_map.items()
                                            if k in weather_df.columns})
    # Drop non-feature columns that shouldn't merge
    drop_cols = [c for c in ["venue_lat", "venue_lon", "kickoff_hour",
                             "weather_date", "weather_code"] if c in weather_df.columns]
    if drop_cols:
        weather_df = weather_df.drop(columns=drop_cols)

    # Merge weather data
    df = _merge_external_data(df, weather_df, "weather")

    # Compute derived weather features
    if "temperature" in df.columns:
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        df["is_hot"] = (df["temperature"] > 32).astype(float)
        df["is_cold"] = (df["temperature"] < 10).astype(float)
        df["temp_deviation"] = (df["temperature"] - 22).abs()

    if "rain_mm" in df.columns:
        df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce")
        df["is_raining"] = (df["rain_mm"] > 0.2).astype(float)
        df["is_heavy_rain"] = (df["rain_mm"] > 2.0).astype(float)

    if "wind_speed" in df.columns or "wind_speed_kmh" in df.columns:
        wind_col = "wind_speed_kmh" if "wind_speed_kmh" in df.columns else "wind_speed"
        df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce")
        df["is_windy"] = (df[wind_col] > 25).astype(float)
        df["is_very_windy"] = (df[wind_col] > 35).astype(float)
        # Rename to standard
        if wind_col != "wind_speed_kmh":
            df["wind_speed_kmh"] = df[wind_col]

    if "humidity" in df.columns or "humidity_pct" in df.columns:
        hum_col = "humidity_pct" if "humidity_pct" in df.columns else "humidity"
        df[hum_col] = pd.to_numeric(df[hum_col], errors="coerce")
        if hum_col != "humidity_pct":
            df["humidity_pct"] = df[hum_col]

    # Interaction: compound bad conditions
    if "rain_mm" in df.columns and "wind_speed_kmh" in df.columns:
        df["rain_x_wind"] = df["rain_mm"].fillna(0) * df["wind_speed_kmh"].fillna(0) / 100.0

    # Load ground conditions from metadata
    meta_df = _try_load_parquet(PROCESSED_DIR / "match_metadata.parquet", "match_metadata (ground)")
    if not meta_df.empty:
        df = _merge_ground_conditions(df, meta_df)

    n_weather = sum(1 for c in ["temperature", "rain_mm", "wind_speed_kmh",
                                "is_raining", "is_heavy_rain", "is_windy",
                                "rain_x_wind", "temp_deviation",
                                "ground_wet", "ground_heavy"]
                    if c in df.columns and df[c].notna().any())
    print(f"  Added {n_weather} weather features")
    return df


def _merge_ground_conditions(matches, meta_df):
    """Extract ground condition features from match_metadata."""
    df = matches.copy()

    if "groundConditions" not in meta_df.columns:
        return df

    meta = meta_df.copy()
    wet_conditions = {"Wet", "Heavy", "Soft", "wet", "heavy", "soft"}

    meta["ground_wet"] = meta["groundConditions"].astype(str).apply(
        lambda x: 1 if x.strip() in wet_conditions else 0
    ).astype(float)
    meta["ground_heavy"] = meta["groundConditions"].astype(str).apply(
        lambda x: 1 if x.strip().lower() == "heavy" else 0
    ).astype(float)

    # Merge onto matches
    cols_to_merge = ["ground_wet", "ground_heavy"]
    # Copy over key columns
    for c in ["year", "round_slug", "home_slug", "away_slug", "round",
              "home_team", "away_team"]:
        if c in meta.columns:
            continue

    df = _merge_external_data(df, meta[
        [c for c in meta.columns if c in ["year", "round_slug", "home_slug",
         "away_slug", "round", "home_team", "away_team"]] + cols_to_merge
    ], "ground_cond")

    return df


# =========================================================================
# V5-FINAL FEATURE: REFEREE
# =========================================================================

def compute_referee_features(matches):
    """Compute referee-based features from match_metadata."""
    print("\n" + "=" * 80)
    print("  V5-FINAL: COMPUTING REFEREE FEATURES")
    print("=" * 80)

    df = matches.copy().sort_values("date").reset_index(drop=True)

    # Check if referee data already exists on matches
    has_referee = "referee" in df.columns and df["referee"].notna().any()

    if not has_referee:
        # Try loading from match_metadata
        meta_df = _try_load_parquet(PROCESSED_DIR / "match_metadata.parquet", "match_metadata (referee)")
        if not meta_df.empty and "referee_name" in meta_df.columns:
            # Merge referee_name onto matches
            df = _merge_external_data(
                df,
                meta_df[[c for c in meta_df.columns
                         if c in ["year", "round_slug", "home_slug", "away_slug",
                                  "round", "home_team", "away_team", "referee_name"]]],
                "referee_meta"
            )
            if "referee_name" in df.columns:
                df["referee"] = df["referee_name"]

    has_referee = "referee" in df.columns and df["referee"].notna().any()
    if not has_referee:
        print("  No referee data available, skipping")
        return df

    # Accumulators for walk-forward referee features
    ref_home_wins = defaultdict(int)
    ref_games = defaultdict(int)
    ref_total_points = defaultdict(float)
    ref_penalties = defaultdict(float)

    ref_home_wr_list = []
    ref_exp_list = []
    ref_avg_total_list = []
    ref_penalty_rate_list = []

    for idx, row in df.iterrows():
        ref = str(row.get("referee", "")).strip()
        hs = row.get("home_score")
        ays = row.get("away_score")
        pen_h = row.get("penalty_home", np.nan)
        pen_a = row.get("penalty_away", np.nan)

        if not ref or ref in ("nan", "None", ""):
            ref_home_wr_list.append(np.nan)
            ref_exp_list.append(np.nan)
            ref_avg_total_list.append(np.nan)
            ref_penalty_rate_list.append(np.nan)
            continue

        games = ref_games[ref]

        # Pre-match features (use walk-forward, so only prior data)
        if games >= 5:
            ref_home_wr_list.append(ref_home_wins[ref] / games)
            ref_exp_list.append(min(games / 100.0, 1.0))
            ref_avg_total_list.append(ref_total_points[ref] / games)
            ref_penalty_rate_list.append(ref_penalties[ref] / games)
        else:
            ref_home_wr_list.append(np.nan)
            ref_exp_list.append(np.nan)
            ref_avg_total_list.append(np.nan)
            ref_penalty_rate_list.append(np.nan)

        # Post-match update
        if pd.notna(hs) and pd.notna(ays):
            ref_games[ref] += 1
            if float(hs) > float(ays):
                ref_home_wins[ref] += 1
            ref_total_points[ref] += float(hs) + float(ays)
            if pd.notna(pen_h) and pd.notna(pen_a):
                ref_penalties[ref] += float(pen_h) + float(pen_a)

    df["referee_avg_total_points"] = ref_avg_total_list
    df["referee_home_win_rate"] = ref_home_wr_list
    df["referee_experience"] = ref_exp_list
    df["referee_penalty_rate"] = ref_penalty_rate_list

    n_valid = df["referee_home_win_rate"].notna().sum()
    print(f"  Added referee features ({n_valid}/{len(df)} valid)")
    return df


# =========================================================================
# V5-FINAL FEATURE: TRAVEL DISTANCE
# =========================================================================

def compute_travel_features(matches):
    """Compute travel distance features from parquet or team state mapping."""
    print("\n" + "=" * 80)
    print("  V5-FINAL: COMPUTING TRAVEL DISTANCE FEATURES")
    print("=" * 80)

    df = matches.copy()

    # Try loading pre-computed travel data
    travel_df = _try_load_parquet(PROCESSED_DIR / "travel_data.parquet", "travel_data")

    if not travel_df.empty:
        df = _merge_external_data(df, travel_df, "travel")
        # Ensure expected columns exist after merge
        if "home_travel_km" in df.columns:
            df["away_long_travel"] = (
                pd.to_numeric(df.get("away_travel_km", 0), errors="coerce") > 1000
            ).astype(float)
            n_valid = df["home_travel_km"].notna().sum()
            print(f"  Travel data merged ({n_valid}/{len(df)} matches)")
            return df

    # Fall back to computed travel from state mapping
    print("  Computing travel from team state mapping...")

    is_interstate = []
    home_travel = []
    away_travel = []
    travel_diff = []

    for _, row in df.iterrows():
        home = row.get("home_team", "")
        away = row.get("away_team", "")

        h_state = TEAM_STATE.get(home, "NSW")
        a_state = TEAM_STATE.get(away, "NSW")

        dist = get_travel_distance(h_state, a_state)
        is_inter = 1 if h_state != a_state else 0

        is_interstate.append(is_inter)
        home_travel.append(0)  # Home team plays at home
        away_travel.append(dist)
        travel_diff.append(-dist)  # Negative = away team has more travel

    df["home_travel_km"] = home_travel
    df["away_travel_km"] = away_travel
    df["travel_diff_km"] = travel_diff
    df["is_interstate"] = is_interstate
    df["away_long_travel"] = (pd.Series(away_travel) > 1000).astype(float).values

    n_inter = sum(is_interstate)
    print(f"  Interstate matches: {n_inter}/{len(df)} ({100*n_inter/len(df):.1f}%)")
    return df


# =========================================================================
# V5-FINAL FEATURE: ENHANCED MATCH STATS ROLLING
# =========================================================================

def compute_enhanced_match_stats_features(matches):
    """Load and merge rolling match stats features from existing parquets."""
    print("\n" + "=" * 80)
    print("  V5-FINAL: COMPUTING ENHANCED MATCH STATS FEATURES")
    print("=" * 80)

    df = matches.copy()

    # Try the enhanced match stats first
    enhanced_path = PROCESSED_DIR / "match_stats_enhanced.parquet"
    enhanced_df = _try_load_parquet(enhanced_path, "match_stats_enhanced")

    # Fall back to regular match_stats_rolling
    rolling_path = PROCESSED_DIR / "match_stats_rolling.parquet"
    rolling_df = _try_load_parquet(rolling_path, "match_stats_rolling")

    if rolling_df.empty and enhanced_df.empty:
        print("  No match stats data available, skipping")
        return df

    stats_df = enhanced_df if not enhanced_df.empty else pd.DataFrame()

    # If we have rolling stats, merge them using the swap-aware merge
    if not rolling_df.empty:
        if "date" in rolling_df.columns:
            rolling_df["date"] = pd.to_datetime(rolling_df["date"], errors="coerce")
        rolling_df["year"] = pd.to_numeric(rolling_df["year"], errors="coerce").astype(int)

        feature_cols_rolling = [c for c in rolling_df.columns
                                if c.startswith(("home_rolling_", "away_rolling_", "diff_rolling_"))]

        if feature_cols_rolling:
            merge_df = rolling_df[["year", "round", "home_team", "away_team"] + feature_cols_rolling].copy()
            df = _merge_external_data(df, merge_df, "match_stats_rolling")

    # Select key features to keep (avoid bloat)
    # We keep 5-game and 8-game windows for the most impactful stats
    key_stats = ["completion_rate", "run_metres", "missed_tackles",
                 "line_breaks", "kick_metres", "effective_tackle_pct",
                 "post_contact_metres", "penalties_conceded", "errors",
                 "offloads"]

    kept_cols = []
    for stat in key_stats:
        for w in [5, 8]:
            for prefix in ["home_rolling_", "away_rolling_", "diff_rolling_"]:
                col = f"{prefix}{stat}_{w}"
                if col in df.columns:
                    kept_cols.append(col)

    print(f"  Keeping {len(kept_cols)} match stats rolling features")
    return df


# =========================================================================
# V5-FINAL: FORTRESS & HALFTIME FEATURES (from existing V5)
# =========================================================================

def compute_fortress_features(matches):
    """Compute venue-specific rolling win rates and fortress strength index."""
    print("\n" + "=" * 80)
    print("  V5-FINAL: COMPUTING FORTRESS / VENUE FAMILIARITY FEATURES")
    print("=" * 80)

    df = matches.copy().sort_values("date").reset_index(drop=True)
    windows = [5, 8]

    team_venue_history = defaultdict(list)
    team_home_record = defaultdict(lambda: [0, 0])
    team_away_record = defaultdict(lambda: [0, 0])

    for w in windows:
        df[f"home_venue_wr_L{w}"] = np.nan
        df[f"away_venue_wr_L{w}"] = np.nan
        df[f"venue_wr_diff_L{w}"] = np.nan
    df["home_fortress_strength"] = np.nan
    df["away_fortress_penalty"] = np.nan
    df["home_venue_game_count"] = 0
    df["away_venue_game_count"] = 0

    for idx, row in df.iterrows():
        home = row.get("home_team", "")
        away = row.get("away_team", "")
        venue = str(row.get("venue", "")).strip()
        hs = row.get("home_score")
        ays = row.get("away_score")

        if not venue or venue in ("nan", "None", "") or not home or not away:
            continue

        key_h = (home, venue)
        key_a = (away, venue)

        hist_h = team_venue_history[key_h]
        hist_a = team_venue_history[key_a]

        df.at[idx, "home_venue_game_count"] = len(hist_h)
        df.at[idx, "away_venue_game_count"] = len(hist_a)

        for w in windows:
            if hist_h:
                df.at[idx, f"home_venue_wr_L{w}"] = np.mean(hist_h[-w:])
            if hist_a:
                df.at[idx, f"away_venue_wr_L{w}"] = np.mean(hist_a[-w:])
            if hist_h and hist_a:
                df.at[idx, f"venue_wr_diff_L{w}"] = np.mean(hist_h[-w:]) - np.mean(hist_a[-w:])

        h_rec = team_home_record[home]
        a_rec = team_away_record[away]
        if h_rec[1] >= 5 and len(hist_h) >= 3:
            overall_home_wr = h_rec[0] / h_rec[1]
            df.at[idx, "home_fortress_strength"] = np.mean(hist_h) - overall_home_wr
        if a_rec[1] >= 5 and len(hist_a) >= 3:
            overall_away_wr = a_rec[0] / a_rec[1]
            df.at[idx, "away_fortress_penalty"] = np.mean(hist_a) - overall_away_wr

        if pd.notna(hs) and pd.notna(ays):
            hw = 1.0 if float(hs) > float(ays) else 0.0
            team_venue_history[key_h].append(hw)
            team_venue_history[key_a].append(1.0 - hw)
            team_home_record[home][0] += int(hw)
            team_home_record[home][1] += 1
            team_away_record[away][0] += int(1.0 - hw)
            team_away_record[away][1] += 1

    df["fortress_diff"] = (
        df["home_fortress_strength"].fillna(0) - df["away_fortress_penalty"].fillna(0)
    )

    n_valid = df["home_venue_wr_L5"].notna().sum()
    print(f"  Added fortress features ({n_valid}/{len(df)} valid)")
    return df


def compute_halftime_dominance(matches):
    """Compute rolling halftime lead rate."""
    print("\n" + "=" * 80)
    print("  V5-FINAL: COMPUTING HALFTIME DOMINANCE FEATURES")
    print("=" * 80)

    df = matches.copy().sort_values("date").reset_index(drop=True)

    if "halftime_home" not in df.columns:
        print("  No halftime data, skipping")
        return df

    team_ht_history = defaultdict(list)

    home_ht_lead_rate = []
    away_ht_lead_rate = []

    for _, row in df.iterrows():
        home = row.get("home_team", "")
        away = row.get("away_team", "")
        ht_h = row.get("halftime_home")
        ht_a = row.get("halftime_away")

        h_hist = team_ht_history[home]
        a_hist = team_ht_history[away]

        if len(h_hist) >= 3:
            home_ht_lead_rate.append(np.mean(h_hist[-8:]))
        else:
            home_ht_lead_rate.append(np.nan)

        if len(a_hist) >= 3:
            away_ht_lead_rate.append(np.mean(a_hist[-8:]))
        else:
            away_ht_lead_rate.append(np.nan)

        if pd.notna(ht_h) and pd.notna(ht_a):
            team_ht_history[home].append(1.0 if float(ht_h) > float(ht_a) else 0.0)
            team_ht_history[away].append(1.0 if float(ht_a) > float(ht_h) else 0.0)

    df["home_ht_lead_rate_8"] = home_ht_lead_rate
    df["away_ht_lead_rate_8"] = away_ht_lead_rate
    df["ht_lead_rate_diff"] = (
        pd.Series(home_ht_lead_rate) - pd.Series(away_ht_lead_rate)
    ).values

    n_valid = df["home_ht_lead_rate_8"].notna().sum()
    print(f"  Added halftime dominance features ({n_valid}/{len(df)} valid)")
    return df


# =========================================================================
# BUILD V5-FINAL FEATURE MATRIX
# =========================================================================

def build_v5_final_feature_matrix(df):
    """Build the comprehensive V5-final feature matrix."""
    print("\n" + "=" * 80)
    print("  BUILDING V5-FINAL FEATURE MATRIX")
    print("=" * 80)

    df["home_win"] = np.where(
        df["home_score"] > df["away_score"], 1.0,
        np.where(df["home_score"] < df["away_score"], 0.0, np.nan)
    )

    feature_cols = []

    # === V3 BASE FEATURES ===
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

    # === V4 FEATURES ===
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

    # V4 Engineered Interactions
    feature_cols += [
        "is_early_season", "is_mid_season", "is_late_season",
        "elo_diff_x_late", "elo_diff_x_early", "form_x_late",
        "home_defense_improving", "away_defense_improving", "defense_trend_diff",
        "scoring_env_ratio", "fav_consistency",
        "elo_spread_agree", "strong_team_rested", "home_ground_x_form",
    ]

    # === V5-FINAL NEW FEATURES ===

    # Player Quality
    feature_cols += [
        "home_team_avg_fantasy_5", "away_team_avg_fantasy_5",
        "home_spine_quality_5", "away_spine_quality_5",
        "home_forward_run_metres_5", "away_forward_run_metres_5",
        "team_fantasy_diff_5", "spine_quality_diff_5",
    ]

    # Weather
    feature_cols += [
        "temperature", "rain_mm", "wind_speed_kmh", "humidity_pct",
        "is_raining", "is_heavy_rain", "is_windy", "is_very_windy",
        "is_hot", "is_cold", "rain_x_wind", "temp_deviation",
        "ground_wet", "ground_heavy",
    ]

    # Referee
    feature_cols += [
        "referee_avg_total_points", "referee_home_win_rate",
        "referee_experience", "referee_penalty_rate",
    ]

    # Travel
    feature_cols += [
        "home_travel_km", "away_travel_km", "travel_diff_km",
        "is_interstate", "away_long_travel",
    ]

    # Fortress / Venue Familiarity
    for w in [5, 8]:
        feature_cols += [
            f"home_venue_wr_L{w}", f"away_venue_wr_L{w}", f"venue_wr_diff_L{w}",
        ]
    feature_cols += [
        "home_fortress_strength", "away_fortress_penalty", "fortress_diff",
        "home_venue_game_count", "away_venue_game_count",
    ]

    # Halftime Dominance
    feature_cols += ["home_ht_lead_rate_8", "away_ht_lead_rate_8", "ht_lead_rate_diff"]

    # Match Stats Rolling (key stats at windows 5 and 8)
    key_stats = ["completion_rate", "run_metres", "missed_tackles",
                 "line_breaks", "kick_metres", "effective_tackle_pct",
                 "post_contact_metres", "penalties_conceded", "errors",
                 "offloads"]
    for stat in key_stats:
        for w in [5, 8]:
            feature_cols += [
                f"home_rolling_{stat}_{w}",
                f"away_rolling_{stat}_{w}",
                f"diff_rolling_{stat}_{w}",
            ]

    # Filter to existing columns and de-duplicate
    feature_cols = list(dict.fromkeys(c for c in feature_cols if c in df.columns))

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
    print(f"  Feature columns: {len(feature_cols)}")

    # Count features by category
    v3_count = sum(1 for c in feature_cols if not any(
        tag in c for tag in ["rolling_", "fortress", "venue_wr_L", "venue_game",
                             "interstate", "travel", "referee_", "ht_lead_rate",
                             "fantasy", "spine", "forward_run", "temperature",
                             "rain", "wind", "humidity", "ground_", "is_hot",
                             "is_cold", "is_raining", "is_heavy_rain", "is_windy",
                             "is_very_windy", "temp_deviation", "rain_x_wind",
                             "away_long_travel"]
    ))
    v5_new = len(feature_cols) - v3_count
    print(f"  Approx V3/V4 base: {v3_count}, V5-final new: {v5_new}")

    # Missing summary
    missing = features[feature_cols].isnull().sum()
    missing_pct = missing / len(features) * 100
    high_miss = missing_pct[missing_pct > 50].sort_values(ascending=False)
    if len(high_miss) > 0:
        print(f"\n  Features with >50% missing:")
        for c, pct in high_miss.head(15).items():
            print(f"    {c:<50} {pct:.1f}%")

    return features, feature_cols


# =========================================================================
# ODDS-FREE FEATURE IDENTIFICATION
# =========================================================================

def get_odds_free_features(feature_cols):
    """Return feature columns excluding all odds-related features."""
    odds_free = []
    for c in feature_cols:
        is_odds = any(sub in c.lower() for sub in ODDS_FEATURE_SUBSTRINGS)
        if not is_odds:
            odds_free.append(c)
    return odds_free


# =========================================================================
# WALK-FORWARD BACKTESTING (V5-FINAL)
# =========================================================================

def fill_missing(X_train, X_test):
    """Fill NaN using train medians; boolean flags with 0."""
    bool_cols = {"home_is_back_to_back", "away_is_back_to_back",
                 "home_bye_last_round", "away_bye_last_round",
                 "is_finals", "odds_home_favourite", "is_home", "is_neutral_venue",
                 "odds_spread_agree", "elo_spread_agree",
                 "attendance_high", "is_night_game", "is_afternoon_game", "is_day_game",
                 "is_early_season", "is_mid_season", "is_late_season",
                 "is_interstate", "away_long_travel",
                 "is_raining", "is_heavy_rain", "is_windy", "is_very_windy",
                 "is_hot", "is_cold", "ground_wet", "ground_heavy"}
    medians = X_train.median()
    Xtr = X_train.copy()
    Xte = X_test.copy()
    for col in X_train.columns:
        fill_val = 0 if col in bool_cols else medians.get(col, 0)
        Xtr[col] = Xtr[col].fillna(fill_val)
        Xte[col] = Xte[col].fillna(fill_val)
    return Xtr, Xte


def compute_sample_weights(years, decay=SAMPLE_WEIGHT_DECAY):
    max_yr = years.max()
    return decay ** (max_yr - years)


def select_top_features(X_train, y_train, feature_cols, sw, top_n=50):
    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.02,
                               verbosity=0, random_state=42)
    model.fit(X_train, y_train, sample_weight=sw)
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return list(imp.head(top_n).index)


def walk_forward_backtest_v5_final(features, feature_cols):
    """Run V5-final walk-forward backtesting with all model variants."""
    print("\n" + "=" * 80)
    print("  V5-FINAL WALK-FORWARD BACKTESTING")
    print("=" * 80)

    df = features.copy()
    all_results = {}

    odds_free_cols = get_odds_free_features(feature_cols)
    print(f"  All features: {len(feature_cols)}, Odds-free features: {len(odds_free_cols)}")

    # All model names — diverse base models for ensemble
    model_names = [
        # Full feature models (V4 params)
        "XGBoost", "LightGBM", "CatBoost", "LogReg",
        "RandomForest", "ExtraTrees",
        # Feature-selected
        "XGB_top50", "CAT_top50", "LGB_top50", "CAT_top30",
        # Odds-free models
        "XGB_NoOdds", "CAT_NoOdds", "LGB_NoOdds",
        # Diversity: V5-tuned params (different error patterns)
        "XGB_V5tune", "CAT_V5tune",
        # Diversity: HistGradientBoosting (different GBM implementation)
        "HistGBM",
        # Diversity: Recent-only training (last 4 years)
        "XGB_Recent", "CAT_Recent",
        # Diversity: RF on odds-free features
        "RF_NoOdds",
        # Baseline
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

        sw = compute_sample_weights(pd.Series(train_years))

        y_parts.append(y_test)
        odds_parts.append(odds_test)
        year_parts.append(np.full(len(y_test), test_year))

        print(f"\n  Fold {fold_idx+1}: Train <=2013-{train_end} ({len(X_train)}) "
              f"-> Test {test_year} ({len(X_test)})")

        # Feature selection
        top50 = select_top_features(X_train, y_train, feature_cols, sw, top_n=50)
        top30 = top50[:30]
        X_train_top50 = X_train[top50]
        X_test_top50 = X_test[top50]
        X_train_top30 = X_train[top30]
        X_test_top30 = X_test[top30]

        # Odds-free data
        of_cols_available = [c for c in odds_free_cols if c in X_train.columns]
        X_train_noodds = X_train[of_cols_available]
        X_test_noodds = X_test[of_cols_available]

        # --- Odds Implied ---
        model_oof["Odds Implied"].append(np.clip(odds_test, 1e-7, 1-1e-7))

        # --- XGBoost (all features) ---
        m = xgb.XGBClassifier(**BEST_XGB_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["XGBoost"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # --- LightGBM (all features) ---
        m = lgbm.LGBMClassifier(**BEST_LGB_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["LightGBM"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # --- CatBoost (all features) ---
        m = CatBoostClassifier(**BEST_CAT_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["CatBoost"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # --- LogReg ---
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        X_train_sc = np.nan_to_num(X_train_sc, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_sc = np.nan_to_num(X_test_sc, nan=0.0, posinf=0.0, neginf=0.0)
        m = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        m.fit(X_train_sc, y_train, sample_weight=sw)
        model_oof["LogReg"].append(np.clip(m.predict_proba(X_test_sc)[:, 1], 1e-7, 1-1e-7))

        # --- Random Forest ---
        m = RandomForestClassifier(**BEST_RF_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["RandomForest"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # --- ExtraTrees ---
        m = ExtraTreesClassifier(**BEST_RF_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["ExtraTrees"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # --- Feature-selected models ---
        m = xgb.XGBClassifier(**BEST_XGB_PARAMS)
        m.fit(X_train_top50, y_train, sample_weight=sw)
        model_oof["XGB_top50"].append(np.clip(m.predict_proba(X_test_top50)[:, 1], 1e-7, 1-1e-7))

        m = lgbm.LGBMClassifier(**BEST_LGB_PARAMS)
        m.fit(X_train_top50, y_train, sample_weight=sw)
        model_oof["LGB_top50"].append(np.clip(m.predict_proba(X_test_top50)[:, 1], 1e-7, 1-1e-7))

        m = CatBoostClassifier(**BEST_CAT_PARAMS)
        m.fit(X_train_top50, y_train, sample_weight=sw)
        model_oof["CAT_top50"].append(np.clip(m.predict_proba(X_test_top50)[:, 1], 1e-7, 1-1e-7))

        m = CatBoostClassifier(**BEST_CAT_PARAMS)
        m.fit(X_train_top30, y_train, sample_weight=sw)
        model_oof["CAT_top30"].append(np.clip(m.predict_proba(X_test_top30)[:, 1], 1e-7, 1-1e-7))

        # --- Odds-free models ---
        m = xgb.XGBClassifier(**BEST_XGB_PARAMS)
        m.fit(X_train_noodds, y_train, sample_weight=sw)
        model_oof["XGB_NoOdds"].append(np.clip(m.predict_proba(X_test_noodds)[:, 1], 1e-7, 1-1e-7))

        m = CatBoostClassifier(**BEST_CAT_PARAMS)
        m.fit(X_train_noodds, y_train, sample_weight=sw)
        model_oof["CAT_NoOdds"].append(np.clip(m.predict_proba(X_test_noodds)[:, 1], 1e-7, 1-1e-7))

        m = lgbm.LGBMClassifier(**BEST_LGB_PARAMS)
        m.fit(X_train_noodds, y_train, sample_weight=sw)
        model_oof["LGB_NoOdds"].append(np.clip(m.predict_proba(X_test_noodds)[:, 1], 1e-7, 1-1e-7))

        # --- Diversity: V5-tuned params (different bias-variance from V4 params) ---
        m = xgb.XGBClassifier(**V5_XGB_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["XGB_V5tune"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        m = CatBoostClassifier(**V5_CAT_PARAMS)
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["CAT_V5tune"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # --- Diversity: HistGradientBoosting (sklearn native GBM) ---
        m = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.02,
            min_samples_leaf=30, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42,
        )
        m.fit(X_train, y_train, sample_weight=sw)
        model_oof["HistGBM"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

        # --- Diversity: Recent-only training (last 4 years only) ---
        recent_mask = df["year"].between(train_end - 3, train_end)
        if recent_mask.sum() >= 200:
            X_tr_recent = df.loc[recent_mask, feature_cols].copy()
            y_tr_recent = df.loc[recent_mask, "home_win"].values
            X_tr_recent, _ = fill_missing(X_tr_recent, X_test_raw)
            sw_recent = compute_sample_weights(pd.Series(df.loc[recent_mask, "year"].values))

            m = xgb.XGBClassifier(**BEST_XGB_PARAMS)
            m.fit(X_tr_recent, y_tr_recent, sample_weight=sw_recent)
            model_oof["XGB_Recent"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))

            m = CatBoostClassifier(**BEST_CAT_PARAMS)
            m.fit(X_tr_recent, y_tr_recent, sample_weight=sw_recent)
            model_oof["CAT_Recent"].append(np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7))
        else:
            # Fall back to full training if not enough recent data
            model_oof["XGB_Recent"].append(model_oof["XGBoost"][-1])
            model_oof["CAT_Recent"].append(model_oof["CatBoost"][-1])

        # --- Diversity: RF on odds-free features ---
        m = RandomForestClassifier(**BEST_RF_PARAMS)
        m.fit(X_train_noodds, y_train, sample_weight=sw)
        model_oof["RF_NoOdds"].append(np.clip(m.predict_proba(X_test_noodds)[:, 1], 1e-7, 1-1e-7))

        # Print fold results
        for n in ["XGBoost", "CatBoost", "XGB_V5tune", "HistGBM", "XGB_Recent", "Odds Implied"]:
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
# V5-FINAL BLENDING, CALIBRATION & STACKING
# =========================================================================

def v5_final_blend_and_stack(all_results, model_oof, y_parts, odds_parts):
    """V5-final: comprehensive blending with calibration, odds-free blends, stacking."""
    print("\n" + "=" * 80)
    print("  V5-FINAL: BLENDING, CALIBRATION & STACKING")
    print("=" * 80)

    all_ml_models = ["XGBoost", "LightGBM", "CatBoost", "LogReg",
                     "XGB_top50", "LGB_top50", "CAT_top50", "CAT_top30",
                     "RandomForest", "ExtraTrees",
                     "XGB_NoOdds", "CAT_NoOdds", "LGB_NoOdds",
                     "XGB_V5tune", "CAT_V5tune", "HistGBM",
                     "XGB_Recent", "CAT_Recent", "RF_NoOdds"]
    active_folds = [i for i in range(len(FOLDS)) if len(y_parts[i]) > 0]

    # === Step 1: Walk-forward calibration (isotonic + Platt) ===
    print("\n  Walk-forward isotonic calibration:")
    calibrated_oof = {}
    platt_oof = {}

    for name in all_ml_models:
        calibrated_oof[name] = [np.array([])] * len(y_parts)
        platt_oof[name] = [np.array([])] * len(y_parts)
        for fold_idx in range(len(y_parts)):
            if len(y_parts[fold_idx]) == 0:
                continue
            cal_X, cal_y = [], []
            for prev in range(fold_idx):
                if len(y_parts[prev]) == 0:
                    continue
                cal_X.append(model_oof[name][prev])
                cal_y.append(y_parts[prev])
            if len(cal_X) < 1:
                calibrated_oof[name][fold_idx] = model_oof[name][fold_idx]
                platt_oof[name][fold_idx] = model_oof[name][fold_idx]
                continue
            cal_X_arr = np.concatenate(cal_X)
            cal_y_arr = np.concatenate(cal_y)
            # Isotonic calibration
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(cal_X_arr, cal_y_arr)
            calibrated_oof[name][fold_idx] = iso.predict(model_oof[name][fold_idx])
            # Platt scaling (sigmoid calibration)
            try:
                platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
                platt.fit(cal_X_arr.reshape(-1, 1), cal_y_arr)
                platt_oof[name][fold_idx] = np.clip(
                    platt.predict_proba(model_oof[name][fold_idx].reshape(-1, 1))[:, 1],
                    1e-7, 1 - 1e-7)
            except Exception:
                platt_oof[name][fold_idx] = model_oof[name][fold_idx]

    calibrated_oof["Odds Implied"] = model_oof["Odds Implied"]
    platt_oof["Odds Implied"] = model_oof["Odds Implied"]

    for name in ["XGBoost", "CatBoost", "XGB_V5tune", "HistGBM", "XGB_NoOdds", "CAT_NoOdds"]:
        if name not in calibrated_oof:
            continue
        orig_lls, cal_lls = [], []
        for i in active_folds:
            if len(y_parts[i]) == 0:
                continue
            orig_lls.append(safe_log_loss(y_parts[i], model_oof[name][i]))
            cal_lls.append(safe_log_loss(y_parts[i], calibrated_oof[name][i]))
        if orig_lls:
            print(f"    {name}: LL {np.mean(orig_lls):.4f} -> {np.mean(cal_lls):.4f}")

    # === Step 2: OptBlend objective function ===
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

    # === Step 3: Multi-model OptBlend combos ===
    print("\n  Multi-Model OptBlend (SLSQP + Nelder-Mead):")

    blend_combos = [
        # Combo 1: 3GBM+Odds
        ("3GBM+Odds",
         ["XGBoost", "LightGBM", "CatBoost"]),
        # Combo 2: All9+Odds (V4 best style)
        ("All9+Odds",
         ["XGBoost", "LightGBM", "CatBoost", "LogReg",
          "XGB_top50", "LGB_top50", "CAT_top50",
          "RandomForest", "ExtraTrees"]),
        # Combo 3: All models (original 13 + 6 new diversity models)
        ("All19+Odds", all_ml_models),
        # Combo 4: Original 13
        ("All13+Odds",
         ["XGBoost", "LightGBM", "CatBoost", "LogReg",
          "XGB_top50", "LGB_top50", "CAT_top50", "CAT_top30",
          "RandomForest", "ExtraTrees",
          "XGB_NoOdds", "CAT_NoOdds", "LGB_NoOdds"]),
        # Combo 5: NoOdds models + RF_NoOdds
        ("NoOdds4+Odds",
         ["XGB_NoOdds", "CAT_NoOdds", "LGB_NoOdds", "RF_NoOdds"]),
        # Combo 6: Best5+Odds
        ("Best5+Odds",
         ["XGBoost", "CatBoost", "XGB_top50", "LGB_top50", "RandomForest"]),
        # Combo 7: Diverse7 (one of each algo/param combo)
        ("Diverse7+Odds",
         ["XGBoost", "CatBoost", "HistGBM", "RandomForest",
          "XGB_V5tune", "CAT_V5tune", "RF_NoOdds"]),
        # Combo 8: Recent+Standard blend
        ("Recent+Std+Odds",
         ["XGBoost", "CatBoost", "XGB_Recent", "CAT_Recent",
          "RandomForest"]),
        # Combo 9: GBM+RF+Odds
        ("GBM+RF+Odds",
         ["XGBoost", "CatBoost", "RandomForest"]),
    ]

    for use_cal, cal_label in [(False, "Raw"), (True, "Cal"), (True, "Platt")]:
        if cal_label == "Platt":
            oof_source = platt_oof
        elif use_cal:
            oof_source = calibrated_oof
        else:
            oof_source = model_oof
        for combo_name, combo in blend_combos:
            n_m = len(combo)

            # Try SLSQP with bounds first (allows negative weights)
            bounds = [(-1.0, 1.0)] * n_m
            x0 = np.array([0.1 / n_m] * n_m)

            try:
                res_slsqp = minimize(
                    multi_blend_obj, x0,
                    args=(combo, oof_source, odds_parts, y_parts, active_folds),
                    method="SLSQP",
                    bounds=bounds,
                    options={"maxiter": 2000, "ftol": 1e-10}
                )
            except Exception:
                res_slsqp = None

            # Also try Nelder-Mead
            res_nm = minimize(
                multi_blend_obj, x0,
                args=(combo, oof_source, odds_parts, y_parts, active_folds),
                method="Nelder-Mead",
                options={"maxiter": 15000, "xatol": 0.0003, "fatol": 1e-9}
            )

            # Pick the better one
            best_res = res_nm
            if res_slsqp is not None and res_slsqp.fun < res_nm.fun:
                best_res = res_slsqp

            bw = best_res.x
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
            label = f"{cal_label}-OptBlend {combo_name} ({weight_str}, odds={w_odds:.3f})"
            all_results[label] = result
            print(f"    {cal_label}-OptBlend {combo_name}: "
                  f"Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    # === Step 4: Walk-Forward OptBlend (no look-ahead in weights) ===
    print("\n  Walk-Forward OptBlend (no look-ahead):")
    wf_combos = [
        ("WF-All19+Odds", all_ml_models),
        ("WF-Diverse7+Odds", ["XGBoost", "CatBoost", "HistGBM", "RandomForest",
                              "XGB_V5tune", "CAT_V5tune", "RF_NoOdds"]),
    ]

    for combo_name, combo in wf_combos:
        wf_probs = [np.array([])] * len(FOLDS)
        for fold_idx in range(len(FOLDS)):
            if len(y_parts[fold_idx]) == 0:
                continue

            if fold_idx < 2:
                n_m = len(combo)
                equal_w = 0.15 / n_m
                blended = np.zeros_like(y_parts[fold_idx], dtype=float)
                for n in combo:
                    blended += equal_w * model_oof[n][fold_idx]
                blended += (1 - 0.15) * odds_parts[fold_idx]
            else:
                prior_folds = [p for p in range(fold_idx) if len(y_parts[p]) > 0]
                n_m = len(combo)
                x0 = np.array([0.05 / n_m] * n_m)
                bounds = [(-1.0, 1.0)] * n_m
                try:
                    res = minimize(
                        multi_blend_obj, x0,
                        args=(combo, model_oof, odds_parts, y_parts, prior_folds),
                        method="SLSQP",
                        bounds=bounds,
                        options={"maxiter": 2000, "ftol": 1e-9}
                    )
                except Exception:
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
            if len(wf_probs[i]) == 0:
                continue
            fold_metrics.append(compute_metrics(y_parts[i], wf_probs[i]))
        if fold_metrics:
            result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            label = f"WF-OptBlend {combo_name}"
            all_results[label] = result
            print(f"    {label}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    # === Step 5: Stacking with LR ===
    print("\n  LR Stacking:")
    for stack_label, stack_models in [
        ("LR-All13+Odds", all_ml_models),
    ]:
        stack_probs = [np.array([])] * len(FOLDS)
        for fold_idx in range(len(FOLDS)):
            if len(y_parts[fold_idx]) == 0:
                continue

            meta_X_train_parts, meta_y_train_parts = [], []
            for prev in range(fold_idx):
                if len(y_parts[prev]) == 0:
                    continue
                row = np.column_stack(
                    [model_oof[n][prev] for n in stack_models] + [odds_parts[prev]]
                )
                meta_X_train_parts.append(row)
                meta_y_train_parts.append(y_parts[prev])

            meta_X_test = np.column_stack(
                [model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]]
            )

            if len(meta_X_train_parts) < 1:
                avg = np.mean(
                    [model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]],
                    axis=0
                )
                stack_probs[fold_idx] = np.clip(avg, 1e-7, 1-1e-7)
                continue

            meta_X_train = np.vstack(meta_X_train_parts)
            meta_y_train = np.concatenate(meta_y_train_parts)

            meta_lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            meta_lr.fit(meta_X_train, meta_y_train)
            stack_probs[fold_idx] = np.clip(
                meta_lr.predict_proba(meta_X_test)[:, 1], 1e-7, 1-1e-7
            )

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

    # === Step 6: Stacking with LightGBM meta-learner ===
    print("\n  LightGBM Stacking:")
    for stack_label, stack_models in [
        ("LGB-All13+Odds", all_ml_models),
    ]:
        stack_probs = [np.array([])] * len(FOLDS)
        for fold_idx in range(len(FOLDS)):
            if len(y_parts[fold_idx]) == 0:
                continue

            meta_X_train_parts, meta_y_train_parts = [], []
            for prev in range(fold_idx):
                if len(y_parts[prev]) == 0:
                    continue
                row = np.column_stack(
                    [model_oof[n][prev] for n in stack_models] + [odds_parts[prev]]
                )
                meta_X_train_parts.append(row)
                meta_y_train_parts.append(y_parts[prev])

            meta_X_test = np.column_stack(
                [model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]]
            )

            if len(meta_X_train_parts) < 2:
                avg = np.mean(
                    [model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]],
                    axis=0
                )
                stack_probs[fold_idx] = np.clip(avg, 1e-7, 1-1e-7)
                continue

            meta_X_train = np.vstack(meta_X_train_parts)
            meta_y_train = np.concatenate(meta_y_train_parts)

            meta_lgb = lgbm.LGBMClassifier(
                n_estimators=100, num_leaves=8, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=20,
                random_state=42, verbose=-1,
            )
            meta_lgb.fit(meta_X_train, meta_y_train)
            stack_probs[fold_idx] = np.clip(
                meta_lgb.predict_proba(meta_X_test)[:, 1], 1e-7, 1-1e-7
            )

        fold_metrics = []
        for i in active_folds:
            if len(stack_probs[i]) == 0:
                continue
            fold_metrics.append(compute_metrics(y_parts[i], stack_probs[i]))
        if fold_metrics:
            result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            label = f"Stacking ({stack_label} -> LGB)"
            all_results[label] = result
            print(f"    {label}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    # === Step 7: MLP Neural Network Stacking ===
    print("\n  MLP Stacking:")
    for stack_label, stack_models in [
        ("MLP-All9+Odds",
         ["XGBoost", "LightGBM", "CatBoost", "LogReg",
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
                row = np.column_stack(
                    [model_oof[n][prev] for n in stack_models] + [odds_parts[prev]]
                )
                meta_X_train_parts.append(row)
                meta_y_train_parts.append(y_parts[prev])

            meta_X_test = np.column_stack(
                [model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]]
            )

            if len(meta_X_train_parts) < 2:
                avg = np.mean(
                    [model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]],
                    axis=0
                )
                stack_probs[fold_idx] = np.clip(avg, 1e-7, 1-1e-7)
                continue

            meta_X_train = np.vstack(meta_X_train_parts)
            meta_y_train = np.concatenate(meta_y_train_parts)

            meta_scaler = StandardScaler()
            meta_X_train_sc = meta_scaler.fit_transform(meta_X_train)
            meta_X_test_sc = meta_scaler.transform(meta_X_test)

            mlp = MLPClassifier(
                hidden_layer_sizes=(32, 16), activation="relu",
                max_iter=500, random_state=42, early_stopping=True,
                validation_fraction=0.15, learning_rate_init=0.001,
            )
            mlp.fit(meta_X_train_sc, meta_y_train)
            stack_probs[fold_idx] = np.clip(
                mlp.predict_proba(meta_X_test_sc)[:, 1], 1e-7, 1-1e-7
            )

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

    return all_results


# =========================================================================
# RESULTS COMPARISON
# =========================================================================

def print_comparison(all_results):
    """Print comprehensive comparison with all baselines."""
    print("\n" + "=" * 80)
    print("  V5-FINAL COMPREHENSIVE RESULTS COMPARISON")
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
    v3_best_ll = 0.5977
    v4_best_ll = 0.5960

    print()
    hdr = (f"{'#':>3}  {'Model':<75} | {'Acc':>7} | {'LL':>7} | {'Brier':>7} | "
           f"{'AUC':>7} | {'vs Odds':>8} | {'vs V4':>8}")
    print(hdr)
    print("-" * len(hdr))

    for idx, row in comp_df.head(35).iterrows():
        ll_diff_odds = row["Log Loss"] - odds_ll
        ll_diff_v4 = row["Log Loss"] - v4_best_ll
        marker = ""
        if row["Model"] == "Odds Implied":
            marker = " <-ODDS"
        elif row["Log Loss"] < v4_best_ll:
            marker = " ***V5>"

        print(
            f"{idx+1:3d}  {row['Model']:<75} | {row['Accuracy']:7.4f} | "
            f"{row['Log Loss']:7.4f} | {row['Brier']:7.4f} | {row['AUC']:7.4f} | "
            f"{ll_diff_odds:+8.4f} | {ll_diff_v4:+8.4f}{marker}"
        )

    print("-" * len(hdr))

    best = comp_df.iloc[0]
    print(f"\n  BEST V5-FINAL MODEL: {best['Model']}")
    print(f"    Accuracy:  {best['Accuracy']:.4f}  (V4 best: ~0.681, odds: {odds_acc:.4f})")
    print(f"    Log Loss:  {best['Log Loss']:.4f}  (V4 best: {v4_best_ll:.4f}, "
          f"V3: {v3_best_ll:.4f}, odds: {odds_ll:.4f})")

    if best["Log Loss"] < odds_ll:
        imp = (odds_ll - best["Log Loss"]) / odds_ll * 100
        print(f"\n    >>> BEATS ODDS BASELINE by {imp:.3f}% in log loss <<<")

    if best["Log Loss"] < v4_best_ll:
        imp_v4 = (v4_best_ll - best["Log Loss"]) / v4_best_ll * 100
        print(f"    >>> BEATS V4 BEST by {imp_v4:.3f}% in log loss <<<")
    else:
        gap = best["Log Loss"] - v4_best_ll
        print(f"\n    V4 still leads by {gap:.4f} in log loss")

    beats_odds = comp_df[comp_df["Log Loss"] < odds_ll]
    beats_v4 = comp_df[comp_df["Log Loss"] < v4_best_ll]
    beats_v3 = comp_df[comp_df["Log Loss"] < v3_best_ll]
    print(f"\n    {len(beats_odds)} beat odds, {len(beats_v4)} beat V4, {len(beats_v3)} beat V3")

    return comp_df


# =========================================================================
# FEATURE IMPORTANCE ANALYSIS
# =========================================================================

def analyze_feature_importance(features, feature_cols):
    """Analyze feature importance across all features."""
    print("\n" + "=" * 80)
    print("  V5-FINAL FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    df = features.copy()
    X = df[feature_cols].copy()
    y = df["home_win"].values

    X = X.fillna(X.median())

    m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.02,
                           verbosity=0, random_state=42)
    m.fit(X, y)
    imp = pd.Series(m.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Mark V5-final new features
    v5_tags = ["rolling_", "fortress", "venue_wr_L", "venue_game",
               "interstate", "travel", "referee_", "ht_lead_rate",
               "fantasy", "spine", "forward_run", "temperature",
               "rain", "wind", "humidity", "ground_", "is_hot",
               "is_cold", "is_raining", "is_heavy_rain", "is_windy",
               "is_very_windy", "temp_deviation", "rain_x_wind",
               "away_long_travel", "NoOdds"]

    print("\n  Top 40 features:")
    for i, (feat, val) in enumerate(imp.head(40).items()):
        marker = ""
        if any(tag in feat for tag in v5_tags):
            marker = " [V5-NEW]"
        print(f"    {i+1:2d}. {feat:<50} {val:.4f}{marker}")

    # Show V5-specific features
    v5_cols = [c for c in feature_cols if any(tag in c for tag in v5_tags)]
    if v5_cols:
        print(f"\n  V5-final new feature importances ({len(v5_cols)} features):")
        v5_imp = imp[imp.index.isin(v5_cols)].sort_values(ascending=False)
        for c, val in v5_imp.head(20).items():
            print(f"    {c:<50} {val:.4f}")

    return imp


# =========================================================================
# MAIN PIPELINE
# =========================================================================

def main():
    overall_start = time.time()

    print()
    print("*" * 80)
    print("*  NRL MATCH PREDICTION - V5 FINAL PIPELINE")
    print("*  All data sources: player quality, weather, referee, travel, match stats")
    print("*  Odds-decorrelated models + LightGBM stacking meta-learner")
    print("*  Goal: Beat V4 best (0.5960 LL) and approach 70% accuracy")
    print("*" * 80)
    print()

    # === STEP 1: Load data ===
    matches, ladders, odds = v3.load_and_fix_homeaway()

    # === STEP 2: Link odds ===
    matches = v3.link_odds(matches, odds)

    # === STEP 3: Tune Elo ===
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

    # === STEP 6: V4 features ===
    matches = v4.compute_v4_odds_features(matches)
    matches = v4.compute_scoring_consistency_features(matches)
    matches = v4.compute_attendance_features(matches)
    matches = v4.compute_kickoff_features(matches)
    matches = v4.compute_v4_engineered_features(matches)

    # === STEP 7: V5-FINAL NEW features ===

    # 7a. Player quality
    matches = compute_player_quality_features(matches)

    # 7b. Weather
    matches = compute_weather_features(matches)

    # 7c. Referee
    matches = compute_referee_features(matches)

    # 7d. Travel distance
    matches = compute_travel_features(matches)

    # 7e. Enhanced match stats rolling
    matches = compute_enhanced_match_stats_features(matches)

    # 7f. Fortress / Venue familiarity
    matches = compute_fortress_features(matches)

    # 7g. Halftime dominance
    matches = compute_halftime_dominance(matches)

    # === STEP 8: Build V5-final feature matrix ===
    features, feature_cols = build_v5_final_feature_matrix(matches)

    # Save features
    output_path = FEATURES_DIR / "features_v5_final.parquet"
    features.to_parquet(output_path, index=False)
    print(f"\n  Saved features_v5_final.parquet: {output_path}")
    print(f"  Shape: {features.shape}")

    # === STEP 9: Feature importance analysis ===
    imp = analyze_feature_importance(features, feature_cols)

    # === STEP 10: Walk-forward backtesting ===
    all_results, model_oof, y_parts, odds_parts, year_parts = walk_forward_backtest_v5_final(
        features, feature_cols
    )

    # === STEP 11: Blending, calibration, stacking ===
    all_results = v5_final_blend_and_stack(all_results, model_oof, y_parts, odds_parts)

    # === STEP 12: Print comparison ===
    comp_df = print_comparison(all_results)

    # Save results
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(report_dir / "v5_final_results_comparison.csv", index=False)
    print(f"\n  Results saved to {report_dir / 'v5_final_results_comparison.csv'}")

    elapsed = time.time() - overall_start
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\n" + "=" * 80)
    print("  V5-FINAL PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
