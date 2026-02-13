"""
NRL Match Prediction - Enhanced Feature Engineering & Retraining Pipeline (V3)
===============================================================================
Builds a significantly richer feature matrix by exploiting ALL available data:

  1. Head-to-Head features (rolling H2H win rate, margin for last 3/5/all meetings)
  2. Ladder Home/Away splits (home_won, home_lost, away_won, away_lost from ladders)
  3. Halftime Score features (avg halftime lead, penalty differential)
  4. Odds Market Richness (line, total, open/close movement, odds range, bookmakers)
  5. Venue features (team win rate at venue, avg total score, neutral venue flag)
  6. Elo Hyperparameter Tuning via Optuna (50 trials)
  7. Momentum / Trend features (form momentum, streak, last result)
  8. Sample Weighting (exponential decay favoring recent seasons)

Walk-forward backtesting with XGBoost, LightGBM, CatBoost, Logistic Regression,
Odds-Blended variants, and a stacking ensemble.

Usage:
    python run_enhance_and_retrain.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

# Walk-forward folds
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

# Best hyperparams tuned on v3 features (via tune_v3_models.py)
BEST_XGB_PARAMS = {
    'n_estimators': 204, 'max_depth': 3, 'learning_rate': 0.014763452528270098,
    'subsample': 0.6271921579877064, 'colsample_bytree': 0.5971829404591527,
    'reg_alpha': 5.778786056184469e-07, 'reg_lambda': 1.0925032175320863e-08,
    'min_child_weight': 13, 'gamma': 1.1999299077194732,
    'eval_metric': 'logloss', 'verbosity': 0, 'random_state': 42,
}

BEST_LGB_PARAMS = {
    'n_estimators': 459, 'num_leaves': 22, 'max_depth': 2,
    'learning_rate': 0.009610243619909037, 'subsample': 0.7884007999827868,
    'colsample_bytree': 0.30113824935229067, 'reg_alpha': 4.68712163479738e-05,
    'reg_lambda': 0.0002550909687118317, 'min_child_samples': 13,
    'random_state': 42, 'verbose': -1,
}

BEST_CAT_PARAMS = {
    'iterations': 439, 'depth': 4, 'learning_rate': 0.010601017079142462,
    'l2_leaf_reg': 0.7093032676393299, 'subsample': 0.8031542697264652,
    'colsample_bylevel': 0.42092328685287345, 'min_data_in_leaf': 38,
    'random_seed': 42, 'verbose': 0, 'allow_writing_files': False,
}


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
# STEP 1: Load data and fix home/away
# =========================================================================
def load_and_fix_homeaway():
    """Load raw data and correct home/away designation using odds."""
    print("=" * 80)
    print("  STEP 1: LOADING DATA & FIXING HOME/AWAY")
    print("=" * 80)

    matches_raw = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    ladders = pd.read_parquet(PROCESSED_DIR / "ladders.parquet")
    odds = pd.read_parquet(PROCESSED_DIR / "odds.parquet")

    print(f"  Matches: {len(matches_raw)} rows")
    print(f"  Ladders: {len(ladders)} rows, cols: {list(ladders.columns)}")
    print(f"  Odds:    {len(odds)} rows")

    matches = matches_raw.copy()
    matches["date"] = matches["parsed_date"]
    matches["season"] = matches["year"].astype(int)

    # Save original columns for swapping
    matches["_orig_team1"] = matches["home_team"]
    matches["_orig_team2"] = matches["away_team"]
    matches["_orig_score1"] = matches["home_score"]
    matches["_orig_score2"] = matches["away_score"]
    for col in ["halftime_home", "halftime_away", "penalty_home", "penalty_away"]:
        if col in matches.columns:
            matches[f"_orig_{col}"] = matches[col]

    # Build odds lookup
    odds_ref = odds[["date", "home_team", "away_team"]].copy()
    odds_ref["date"] = pd.to_datetime(odds_ref["date"], errors="coerce")
    odds_set_exact = set()
    for _, orow in odds_ref.iterrows():
        d, h, a = orow["date"], orow["home_team"], orow["away_team"]
        if pd.notna(d):
            odds_set_exact.add((d, h, a))

    matched_order = []
    actual_home, actual_away = [], []
    actual_hscore, actual_ascore = [], []
    n_direct = n_swapped = n_fuzzy = n_unmatched = 0

    for _, row in matches.iterrows():
        dt = row["date"]
        t1, t2 = row["_orig_team1"], row["_orig_team2"]
        s1, s2 = row["_orig_score1"], row["_orig_score2"]
        found = False

        if (dt, t1, t2) in odds_set_exact:
            actual_home.append(t1); actual_away.append(t2)
            actual_hscore.append(s1); actual_ascore.append(s2)
            matched_order.append("direct"); n_direct += 1; found = True
        elif (dt, t2, t1) in odds_set_exact:
            actual_home.append(t2); actual_away.append(t1)
            actual_hscore.append(s2); actual_ascore.append(s1)
            matched_order.append("swapped"); n_swapped += 1; found = True
        else:
            for delta in range(-2, 3):
                if delta == 0:
                    continue
                fuzzy_dt = dt + pd.Timedelta(days=delta)
                if (fuzzy_dt, t1, t2) in odds_set_exact:
                    actual_home.append(t1); actual_away.append(t2)
                    actual_hscore.append(s1); actual_ascore.append(s2)
                    matched_order.append("fuzzy"); n_fuzzy += 1; found = True; break
                elif (fuzzy_dt, t2, t1) in odds_set_exact:
                    actual_home.append(t2); actual_away.append(t1)
                    actual_hscore.append(s2); actual_ascore.append(s1)
                    matched_order.append("fuzzy_swapped"); n_fuzzy += 1; found = True; break

        if not found:
            actual_home.append(t1); actual_away.append(t2)
            actual_hscore.append(s1); actual_ascore.append(s2)
            matched_order.append("unknown"); n_unmatched += 1

    matches["home_team"] = actual_home
    matches["away_team"] = actual_away
    matches["home_score"] = actual_hscore
    matches["away_score"] = actual_ascore
    matches["_match_order"] = matched_order

    # Swap halftime/penalty for swapped matches
    swap_mask = matches["_match_order"].isin(["swapped", "fuzzy_swapped"])
    if "halftime_home" in matches.columns:
        matches.loc[swap_mask, "halftime_home"] = matches.loc[swap_mask, "_orig_halftime_away"]
        matches.loc[swap_mask, "halftime_away"] = matches.loc[swap_mask, "_orig_halftime_home"]
    if "penalty_home" in matches.columns:
        matches.loc[swap_mask, "penalty_home"] = matches.loc[swap_mask, "_orig_penalty_away"]
        matches.loc[swap_mask, "penalty_away"] = matches.loc[swap_mask, "_orig_penalty_home"]

    # Drop unmatched
    matches = matches[matches["_match_order"] != "unknown"].reset_index(drop=True)
    temp_cols = [c for c in matches.columns if c.startswith("_orig_") or c == "_match_order"]
    matches = matches.drop(columns=temp_cols, errors="ignore")

    print(f"  Matched: direct={n_direct}, swapped={n_swapped}, fuzzy={n_fuzzy}, dropped={n_unmatched}")
    hw = (matches["home_score"] > matches["away_score"]).sum()
    aw = (matches["away_score"] > matches["home_score"]).sum()
    print(f"  Home wins: {hw}, Away wins: {aw}")

    # Sort chronologically
    def round_sort_key(r):
        try:
            return int(r)
        except (ValueError, TypeError):
            r_str = str(r).lower()
            if "qualif" in r_str: return 100
            elif "elim" in r_str: return 101
            elif "semi" in r_str: return 102
            elif "prelim" in r_str: return 103
            elif "grand" in r_str: return 104
            return 99
    matches["_rs"] = matches["round"].apply(round_sort_key)
    matches = matches.sort_values(["year", "_rs", "date"]).reset_index(drop=True)
    matches = matches.drop(columns=["_rs"])

    return matches, ladders, odds


# =========================================================================
# STEP 2: Link matches with odds
# =========================================================================
def link_odds(matches, odds):
    """Merge odds columns onto matches."""
    print("\n" + "=" * 80)
    print("  STEP 2: LINKING ODDS DATA")
    print("=" * 80)

    odds_link = odds.copy()
    odds_link["date"] = pd.to_datetime(odds_link["date"], errors="coerce")
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

    exclude = {"date", "home_team", "away_team", "home_score", "away_score",
               "venue", "kickoff", "notes", "home_win"}
    odds_feature_cols = [c for c in odds_link.columns if c not in exclude]

    # Pass 1: exact
    merged = matches.merge(
        odds_link[["date", "home_team", "away_team"] + odds_feature_cols],
        on=["date", "home_team", "away_team"], how="left", indicator=True, suffixes=("", "_odds"),
    )
    exact_mask = merged["_merge"] == "both"
    n_exact = exact_mask.sum()
    exact_df = merged.loc[exact_mask].drop(columns=["_merge"])
    unmatched_df = merged.loc[~exact_mask].copy()

    # Drop odds columns from unmatched
    drop_cols = [c for c in odds_feature_cols if c in unmatched_df.columns]
    drop_cols += [c + "_odds" for c in odds_link.columns if c + "_odds" in unmatched_df.columns]
    drop_cols += ["_merge"]
    drop_cols = [c for c in drop_cols if c in unmatched_df.columns]
    unmatched_df = unmatched_df.drop(columns=drop_cols)

    # Pass 2: fuzzy +/- 2 days
    fuzzy_matched = []
    still_unmatched_idx = []
    odds_for_fuzzy = odds_link[["date", "home_team", "away_team"] + odds_feature_cols].copy()

    for idx, row in unmatched_df.iterrows():
        md = row.get("date")
        if pd.isna(md):
            still_unmatched_idx.append(idx)
            continue
        cand = odds_for_fuzzy.loc[
            (odds_for_fuzzy["home_team"] == row["home_team"]) &
            (odds_for_fuzzy["away_team"] == row["away_team"]) &
            (odds_for_fuzzy["date"] >= md - pd.Timedelta(days=2)) &
            (odds_for_fuzzy["date"] <= md + pd.Timedelta(days=2))
        ]
        if len(cand) >= 1:
            best = cand.copy()
            best["_diff"] = (best["date"] - md).abs()
            best = best.sort_values("_diff").iloc[0]
            combined = row.copy()
            for c in odds_feature_cols:
                if c in best.index:
                    combined[c] = best[c]
            fuzzy_matched.append(combined)
        else:
            still_unmatched_idx.append(idx)

    parts = [exact_df]
    if fuzzy_matched:
        parts.append(pd.DataFrame(fuzzy_matched))
    if still_unmatched_idx:
        parts.append(unmatched_df.loc[still_unmatched_idx])
    matches = pd.concat(parts, ignore_index=True)
    matches = matches.sort_values("date").reset_index(drop=True)

    # Drop _odds suffix columns
    odds_suf = [c for c in matches.columns if c.endswith("_odds")]
    matches = matches.drop(columns=odds_suf, errors="ignore")

    if "season" not in matches.columns:
        matches["season"] = matches["date"].dt.year

    n_with = matches["h2h_home"].notna().sum() if "h2h_home" in matches.columns else 0
    print(f"  Exact: {n_exact}, Fuzzy: {len(fuzzy_matched)}, Unmatched: {len(still_unmatched_idx)}")
    print(f"  With odds: {n_with}/{len(matches)} ({n_with/len(matches)*100:.1f}%)")

    return matches


# =========================================================================
# STEP 3: Elo Tuning via Optuna
# =========================================================================
def tune_elo(matches, n_trials=50):
    """Tune Elo hyperparameters using walk-forward on 2015-2022 data."""
    print("\n" + "=" * 80)
    print("  STEP 3: ELO HYPERPARAMETER TUNING (Optuna, %d trials)" % n_trials)
    print("=" * 80)

    from processing.elo import EloRating

    # Use ALL matches for backfill, evaluate on 2015-2022 window (broader than before)
    all_matches = matches.copy().reset_index(drop=True)

    def elo_objective(trial):
        k = trial.suggest_float("k_factor", 10, 50)
        ha = trial.suggest_float("home_advantage", 20, 80)
        reset = trial.suggest_float("season_reset_factor", 0.4, 1.0)
        mov = trial.suggest_categorical("mov_adjustment", ["none", "linear", "logarithmic"])

        elo = EloRating(
            k_factor=k, home_advantage=ha, season_reset_factor=reset,
            mov_adjustment=mov, mov_linear_divisor=10.0,
        )
        df = elo.backfill(all_matches)
        # Evaluate on 2015-2022 (wider range for more robust tuning, leave 2023+ as holdout)
        mask = (df["year"] >= 2015) & (df["year"] <= 2022)
        sub = df.loc[mask].dropna(subset=["home_elo_prob", "home_score", "away_score"])
        y_true = (sub["home_score"] > sub["away_score"]).astype(float)
        y_prob = sub["home_elo_prob"].values
        return safe_log_loss(y_true, y_prob)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(elo_objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    print(f"  Best Elo params: K={bp['k_factor']:.1f}, HA={bp['home_advantage']:.1f}, "
          f"Reset={bp['season_reset_factor']:.3f}, MOV={bp['mov_adjustment']}")
    print(f"  Best Elo log loss (2015-2022): {study.best_value:.4f}")

    return bp


# =========================================================================
# STEP 4: Backfill Elo with optimal params
# =========================================================================
def backfill_elo(matches, elo_params):
    """Backfill Elo ratings using tuned parameters."""
    print("\n" + "=" * 80)
    print("  STEP 4: BACKFILLING ELO WITH OPTIMAL PARAMS")
    print("=" * 80)

    from processing.elo import EloRating

    elo = EloRating(
        k_factor=elo_params["k_factor"],
        home_advantage=elo_params["home_advantage"],
        season_reset_factor=elo_params["season_reset_factor"],
        mov_adjustment=elo_params["mov_adjustment"],
        mov_linear_divisor=10.0,
    )
    matches = elo.backfill(matches)
    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]

    print(f"  Elo backfilled for {len(matches)} matches")
    return matches


# =========================================================================
# STEP 5: Compute rolling form features
# =========================================================================
def compute_rolling_form_features(matches):
    """Compute rolling form stats (win rate, avg pf/pa/margin) for windows 3, 5, 8."""
    print("\n" + "=" * 80)
    print("  STEP 5: COMPUTING ROLLING FORM FEATURES")
    print("=" * 80)

    windows = [3, 5, 8]
    df = matches.copy()

    # Build team match log
    home_log = pd.DataFrame({
        "match_idx": range(len(df)), "team": df["home_team"], "opponent": df["away_team"],
        "points_for": df["home_score"], "points_against": df["away_score"],
        "halftime_lead": df["halftime_home"].values - df["halftime_away"].values if "halftime_home" in df.columns else np.nan,
        "penalty_for": df.get("penalty_home", pd.Series(np.nan, index=df.index)).values,
        "penalty_against": df.get("penalty_away", pd.Series(np.nan, index=df.index)).values,
        "date": df["date"], "is_home": True,
    })
    away_log = pd.DataFrame({
        "match_idx": range(len(df)), "team": df["away_team"], "opponent": df["home_team"],
        "points_for": df["away_score"], "points_against": df["home_score"],
        "halftime_lead": df["halftime_away"].values - df["halftime_home"].values if "halftime_home" in df.columns else np.nan,
        "penalty_for": df.get("penalty_away", pd.Series(np.nan, index=df.index)).values,
        "penalty_against": df.get("penalty_home", pd.Series(np.nan, index=df.index)).values,
        "date": df["date"], "is_home": False,
    })
    log = pd.concat([home_log, away_log], ignore_index=True)
    log["margin"] = log["points_for"] - log["points_against"]
    log["win"] = np.where(log["margin"] > 0, 1.0, np.where(log["margin"] == 0, 0.5, 0.0))
    log["penalty_diff"] = log["penalty_for"] - log["penalty_against"]
    log = log.sort_values(["team", "date", "match_idx"]).reset_index(drop=True)

    # Build lookup: (team, match_idx) -> {stat_w: value}
    lookup = {}
    for team in log["team"].unique():
        t_log = log[log["team"] == team].copy().reset_index(drop=True)
        for i, row in t_log.iterrows():
            midx = int(row["match_idx"])
            key = (team, midx)
            lookup.setdefault(key, {})

            prior = t_log.iloc[:i]
            if len(prior) == 0:
                for w in windows:
                    for s in ("win_rate", "avg_pf", "avg_pa", "avg_margin"):
                        lookup[key][f"{s}_{w}"] = np.nan
                lookup[key]["avg_halftime_lead_5"] = np.nan
                lookup[key]["avg_penalty_diff_5"] = np.nan
                lookup[key]["streak"] = 0
                lookup[key]["last_result"] = np.nan
                continue

            # Streak and last result
            wins_losses = prior["win"].values
            lookup[key]["last_result"] = wins_losses[-1]
            streak = 0
            last_val = wins_losses[-1]
            for v in reversed(wins_losses):
                if v == last_val:
                    streak += 1
                else:
                    break
            lookup[key]["streak"] = streak if last_val == 1.0 else -streak

            # Rolling windows
            for w in windows:
                pw = prior.tail(w)
                lookup[key][f"win_rate_{w}"] = pw["win"].mean()
                lookup[key][f"avg_pf_{w}"] = pw["points_for"].mean()
                lookup[key][f"avg_pa_{w}"] = pw["points_against"].mean()
                lookup[key][f"avg_margin_{w}"] = pw["margin"].mean()

            # Halftime lead (last 5)
            p5 = prior.tail(5)
            ht_vals = p5["halftime_lead"].dropna()
            lookup[key]["avg_halftime_lead_5"] = ht_vals.mean() if len(ht_vals) > 0 else np.nan

            # Penalty differential (last 5)
            pen_vals = p5["penalty_diff"].dropna()
            lookup[key]["avg_penalty_diff_5"] = pen_vals.mean() if len(pen_vals) > 0 else np.nan

    # Attach to DataFrame
    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        for w in windows:
            for stat in ("win_rate", "avg_pf", "avg_pa", "avg_margin"):
                col_name = f"{side}_{stat}_{w}"
                df[col_name] = [lookup.get((df.iloc[i][team_col], i), {}).get(f"{stat}_{w}", np.nan) for i in range(len(df))]

        # Extra features
        df[f"{side}_avg_halftime_lead_5"] = [lookup.get((df.iloc[i][team_col], i), {}).get("avg_halftime_lead_5", np.nan) for i in range(len(df))]
        df[f"{side}_avg_penalty_diff_5"] = [lookup.get((df.iloc[i][team_col], i), {}).get("avg_penalty_diff_5", np.nan) for i in range(len(df))]
        df[f"{side}_streak"] = [lookup.get((df.iloc[i][team_col], i), {}).get("streak", 0) for i in range(len(df))]
        df[f"{side}_last_result"] = [lookup.get((df.iloc[i][team_col], i), {}).get("last_result", np.nan) for i in range(len(df))]

    # Differentials
    for w in windows:
        df[f"win_rate_diff_{w}"] = df[f"home_win_rate_{w}"] - df[f"away_win_rate_{w}"]
        df[f"avg_margin_diff_{w}"] = df[f"home_avg_margin_{w}"] - df[f"away_avg_margin_{w}"]

    n_cols = sum(1 for c in df.columns if any(p in c for p in ["win_rate_", "avg_pf_", "avg_pa_", "avg_margin_", "halftime", "penalty_diff", "streak", "last_result"]))
    print(f"  Added {n_cols} rolling/form features")
    return df


# =========================================================================
# STEP 6: Head-to-Head features
# =========================================================================
def compute_h2h_features(matches):
    """Compute H2H features: win rate and avg margin for last 3, 5, and all meetings."""
    print("\n" + "=" * 80)
    print("  STEP 6: COMPUTING HEAD-TO-HEAD FEATURES")
    print("=" * 80)

    df = matches.copy()
    lookbacks = [3, 5, "all"]

    h2h_cols = {}
    for lb in lookbacks:
        lb_str = str(lb)
        h2h_cols[f"h2h_home_win_rate_{lb_str}"] = []
        h2h_cols[f"h2h_avg_margin_{lb_str}"] = []
        h2h_cols[f"h2h_matches_{lb_str}"] = []

    for i in range(len(df)):
        home = df.iloc[i]["home_team"]
        away = df.iloc[i]["away_team"]

        # All prior meetings
        prior = df.iloc[:i]
        h2h_prior = prior.loc[
            ((prior["home_team"] == home) & (prior["away_team"] == away)) |
            ((prior["home_team"] == away) & (prior["away_team"] == home))
        ].dropna(subset=["home_score", "away_score"])

        for lb in lookbacks:
            lb_str = str(lb)
            if isinstance(lb, int):
                window = h2h_prior.tail(lb)
            else:
                window = h2h_prior  # "all"

            if len(window) == 0:
                h2h_cols[f"h2h_home_win_rate_{lb_str}"].append(np.nan)
                h2h_cols[f"h2h_avg_margin_{lb_str}"].append(np.nan)
                h2h_cols[f"h2h_matches_{lb_str}"].append(0)
                continue

            margins, wins = [], []
            for _, h_row in window.iterrows():
                if h_row["home_team"] == home:
                    m = h_row["home_score"] - h_row["away_score"]
                else:
                    m = h_row["away_score"] - h_row["home_score"]
                margins.append(m)
                wins.append(1.0 if m > 0 else (0.5 if m == 0 else 0.0))

            h2h_cols[f"h2h_home_win_rate_{lb_str}"].append(np.mean(wins))
            h2h_cols[f"h2h_avg_margin_{lb_str}"].append(np.mean(margins))
            h2h_cols[f"h2h_matches_{lb_str}"].append(len(window))

    for col_name, values in h2h_cols.items():
        df[col_name] = values

    print(f"  Added {len(h2h_cols)} H2H columns")
    return df


# =========================================================================
# STEP 7: Ladder features including home/away splits
# =========================================================================
def compute_ladder_features(matches, ladders):
    """Compute ladder features including home/away splits."""
    print("\n" + "=" * 80)
    print("  STEP 7: COMPUTING LADDER FEATURES (incl. home/away splits)")
    print("=" * 80)

    df = matches.copy()
    lad = ladders.copy()
    lad["round_num"] = pd.to_numeric(lad["round"], errors="coerce")

    # Build lookup: (year, round_num, team) -> ladder stats
    ladder_lookup = {}
    for _, row in lad.iterrows():
        yr = row["year"]
        rn = row.get("round_num")
        team = row["team"]
        if pd.notna(yr) and pd.notna(rn) and pd.notna(team):
            key = (int(yr), int(rn), team)
            ladder_lookup[key] = row.to_dict()

    def get_round_num(r):
        try:
            return int(r)
        except (ValueError, TypeError):
            return None

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        positions, wins_list, losses_list = [], [], []
        pf_vals, pa_vals, pd_vals, cp_vals = [], [], [], []
        hw_vals, hl_vals, aw_vals, al_vals = [], [], [], []
        home_ppg_vals, away_ppg_vals = [], []
        home_pag_vals, away_pag_vals = [], []

        for _, row in df.iterrows():
            yr = row.get("year", row.get("season"))
            r = row.get("round")
            team = row.get(team_col)
            prev_round = get_round_num(r)
            if prev_round is not None:
                prev_round -= 1

            if prev_round is None or prev_round < 1 or pd.isna(yr) or pd.isna(team):
                for lst in [positions, wins_list, losses_list, pf_vals, pa_vals, pd_vals, cp_vals,
                            hw_vals, hl_vals, aw_vals, al_vals, home_ppg_vals, away_ppg_vals,
                            home_pag_vals, away_pag_vals]:
                    lst.append(np.nan)
                continue

            stats = ladder_lookup.get((int(yr), prev_round, team))
            if stats is None:
                for lst in [positions, wins_list, losses_list, pf_vals, pa_vals, pd_vals, cp_vals,
                            hw_vals, hl_vals, aw_vals, al_vals, home_ppg_vals, away_ppg_vals,
                            home_pag_vals, away_pag_vals]:
                    lst.append(np.nan)
            else:
                positions.append(stats.get("position"))
                wins_list.append(stats.get("won"))
                losses_list.append(stats.get("lost"))
                pf_vals.append(stats.get("points_for"))
                pa_vals.append(stats.get("points_against"))
                pd_vals.append(stats.get("points_diff"))
                cp_vals.append(stats.get("competition_points"))
                hw_vals.append(stats.get("home_won"))
                hl_vals.append(stats.get("home_lost"))
                aw_vals.append(stats.get("away_won"))
                al_vals.append(stats.get("away_lost"))

                # PPG at home/away
                hp = stats.get("home_played", 0)
                ap = stats.get("away_played", 0)
                home_ppg_vals.append(stats.get("home_for", 0) / hp if hp > 0 else np.nan)
                away_ppg_vals.append(stats.get("away_for", 0) / ap if ap > 0 else np.nan)
                home_pag_vals.append(stats.get("home_against", 0) / hp if hp > 0 else np.nan)
                away_pag_vals.append(stats.get("away_against", 0) / ap if ap > 0 else np.nan)

        df[f"{side}_ladder_pos"] = positions
        df[f"{side}_wins"] = wins_list
        df[f"{side}_losses"] = losses_list
        df[f"{side}_points_for_season"] = pf_vals
        df[f"{side}_points_against_season"] = pa_vals
        df[f"{side}_points_diff_season"] = pd_vals
        df[f"{side}_competition_points"] = cp_vals
        df[f"{side}_home_won"] = hw_vals
        df[f"{side}_home_lost"] = hl_vals
        df[f"{side}_away_won"] = aw_vals
        df[f"{side}_away_lost"] = al_vals
        df[f"{side}_home_ppg"] = home_ppg_vals
        df[f"{side}_away_ppg"] = away_ppg_vals
        df[f"{side}_home_pag"] = home_pag_vals
        df[f"{side}_away_pag"] = away_pag_vals

    # Derived: home/away win pct from ladder
    for side in ["home", "away"]:
        hw = pd.to_numeric(df[f"{side}_home_won"], errors="coerce")
        hl = pd.to_numeric(df[f"{side}_home_lost"], errors="coerce")
        hp = hw + hl
        df[f"{side}_home_win_pct"] = np.where(hp > 0, hw / hp, np.nan)

        aw = pd.to_numeric(df[f"{side}_away_won"], errors="coerce")
        al = pd.to_numeric(df[f"{side}_away_lost"], errors="coerce")
        ap = aw + al
        df[f"{side}_away_win_pct"] = np.where(ap > 0, aw / ap, np.nan)

    df["ladder_pos_diff"] = pd.to_numeric(df["home_ladder_pos"], errors="coerce") - pd.to_numeric(df["away_ladder_pos"], errors="coerce")

    # KEY: home team's home win pct vs away team's away win pct
    df["home_team_home_win_pct"] = df["home_home_win_pct"]
    df["away_team_away_win_pct"] = df["away_away_win_pct"]
    df["home_team_home_ppg"] = df["home_home_ppg"]
    df["away_team_away_ppg"] = df["away_away_ppg"]

    n_valid = df["home_ladder_pos"].notna().sum()
    print(f"  Ladder data available for {n_valid}/{len(df)} matches")
    print(f"  Added ladder + home/away split features")
    return df


# =========================================================================
# STEP 8: Venue features
# =========================================================================
def compute_venue_features(matches):
    """Compute venue-based features."""
    print("\n" + "=" * 80)
    print("  STEP 8: COMPUTING VENUE FEATURES")
    print("=" * 80)

    df = matches.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Accumulators
    team_venue_record = {}  # (team, venue) -> [wins, games]
    venue_scoring = {}  # venue -> [total_sum, count]
    team_season_venues = {}  # (team, season) -> {venue: count}

    # Known NRL "home" venues per team (for neutral detection)
    is_home_ground_list = []
    home_venue_wr_list = []
    away_venue_wr_list = []
    venue_avg_score_list = []
    is_neutral_list = []

    for _, row in df.iterrows():
        venue = str(row.get("venue", "")).strip()
        home = row.get("home_team", "")
        away = row.get("away_team", "")
        season = row.get("season")
        hs = row.get("home_score")
        ays = row.get("away_score")

        if not venue or venue in ("nan", "None", ""):
            is_home_ground_list.append(np.nan)
            home_venue_wr_list.append(np.nan)
            away_venue_wr_list.append(np.nan)
            venue_avg_score_list.append(np.nan)
            is_neutral_list.append(0)
            continue

        # Is home ground?
        if pd.notna(season) and pd.notna(home):
            sv = team_season_venues.get((home, int(season)), {})
            total_hg = sum(sv.values())
            vg = sv.get(venue, 0)
            is_hg = (vg / total_hg > 0.5) if total_hg >= 3 else np.nan
            is_home_ground_list.append(is_hg)
            # Neutral venue = not home ground for home team and not home ground for away team
            sv_away = team_season_venues.get((away, int(season)), {})
            total_ag = sum(sv_away.values())
            vg_a = sv_away.get(venue, 0)
            if total_hg >= 3 and total_ag >= 3:
                is_neutral_list.append(1 if (vg / total_hg < 0.3 and vg_a / total_ag < 0.3) else 0)
            else:
                is_neutral_list.append(0)
        else:
            is_home_ground_list.append(np.nan)
            is_neutral_list.append(0)

        # Historical win rate at venue (PRE-match)
        h_rec = team_venue_record.get((home, venue), [0, 0])
        a_rec = team_venue_record.get((away, venue), [0, 0])
        home_venue_wr_list.append(h_rec[0] / h_rec[1] if h_rec[1] > 0 else np.nan)
        away_venue_wr_list.append(a_rec[0] / a_rec[1] if a_rec[1] > 0 else np.nan)

        # Avg total at venue
        v_sc = venue_scoring.get(venue, [0.0, 0])
        venue_avg_score_list.append(v_sc[0] / v_sc[1] if v_sc[1] > 0 else np.nan)

        # UPDATE accumulators after recording pre-match features
        if pd.notna(hs) and pd.notna(ays):
            total = float(hs) + float(ays)
            is_hw = float(hs) > float(ays)
            is_aw_w = float(ays) > float(hs)

            rec = team_venue_record.setdefault((home, venue), [0, 0])
            rec[0] += int(is_hw); rec[1] += 1
            rec = team_venue_record.setdefault((away, venue), [0, 0])
            rec[0] += int(is_aw_w); rec[1] += 1

            v = venue_scoring.setdefault(venue, [0.0, 0])
            v[0] += total; v[1] += 1

            if pd.notna(season):
                sv = team_season_venues.setdefault((home, int(season)), {})
                sv[venue] = sv.get(venue, 0) + 1

    df["is_home_ground"] = is_home_ground_list
    df["home_venue_win_rate"] = home_venue_wr_list
    df["away_venue_win_rate"] = away_venue_wr_list
    df["venue_avg_total_score"] = venue_avg_score_list
    df["is_neutral_venue"] = is_neutral_list

    print(f"  Added venue features (is_home_ground, home/away_venue_win_rate, venue_avg_total_score, is_neutral_venue)")
    return df


# =========================================================================
# STEP 9: Odds features (rich)
# =========================================================================
def compute_odds_features(matches):
    """Compute rich odds-derived features."""
    print("\n" + "=" * 80)
    print("  STEP 9: COMPUTING RICH ODDS FEATURES")
    print("=" * 80)

    df = matches.copy()

    # Main implied probs (fair)
    if "h2h_home" in df.columns:
        ip_h = 1.0 / df["h2h_home"]
        ip_a = 1.0 / df["h2h_away"]
        overround = ip_h + ip_a
        df["odds_home_prob"] = ip_h / overround
        df["odds_away_prob"] = ip_a / overround
        df["odds_home_favourite"] = (df["odds_home_prob"] > 0.5).astype(int)
        df["odds_overround"] = overround

    # Opening odds
    if "h2h_home_open" in df.columns:
        ip_ho = 1.0 / df["h2h_home_open"]
        ip_ao = 1.0 / df["h2h_away_open"]
        or_open = ip_ho + ip_ao
        df["odds_home_open_prob"] = ip_ho / or_open
        df["odds_away_open_prob"] = ip_ao / or_open

    # Closing odds
    if "h2h_home_close" in df.columns:
        ip_hc = 1.0 / df["h2h_home_close"]
        ip_ac = 1.0 / df["h2h_away_close"]
        or_close = ip_hc + ip_ac
        df["odds_home_close_prob"] = ip_hc / or_close

        if "odds_home_open_prob" in df.columns:
            df["odds_movement"] = df["odds_home_close_prob"] - df["odds_home_open_prob"]
            df["odds_movement_abs"] = df["odds_movement"].abs()

    # Line/handicap spread (OPENING - safe to use)
    if "line_home_open" in df.columns:
        df["spread_home_open"] = df["line_home_open"]

    # Total line (expected total points)
    if "total_line_open" in df.columns:
        df["total_line_open"] = df["total_line_open"]

    # Odds range (market uncertainty)
    if "h2h_home_max" in df.columns and "h2h_home_min" in df.columns:
        df["odds_home_range"] = df["h2h_home_max"] - df["h2h_home_min"]
        df["odds_away_range"] = df["h2h_away_max"] - df["h2h_away_min"]

    # Bookmakers surveyed
    if "bookmakers_surveyed" in df.columns:
        df["bookmakers_surveyed"] = df["bookmakers_surveyed"]

    n_odds = df["odds_home_prob"].notna().sum() if "odds_home_prob" in df.columns else 0
    print(f"  Odds features computed for {n_odds}/{len(df)} matches")
    return df


# =========================================================================
# STEP 10: Schedule features
# =========================================================================
def compute_schedule_features(matches):
    """Compute schedule/rest features."""
    print("\n" + "=" * 80)
    print("  STEP 10: COMPUTING SCHEDULE FEATURES")
    print("=" * 80)

    df = matches.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Build per-team date list
    team_dates_list = {}
    for i, row in df.iterrows():
        dt = row["date"]
        if pd.isna(dt):
            continue
        for tc in ("home_team", "away_team"):
            team = row[tc]
            if pd.notna(team):
                team_dates_list.setdefault(team, []).append((i, dt))

    for team in team_dates_list:
        team_dates_list[team] = sorted(team_dates_list[team], key=lambda x: x[1])

    team_prev_date = {}
    for team, dl in team_dates_list.items():
        for j, (midx, dt) in enumerate(dl):
            team_prev_date[(team, midx)] = dl[j-1][1] if j > 0 else pd.NaT

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        days_rest, btb, bye = [], [], []
        for i, row in df.iterrows():
            team = row[team_col]
            md = row["date"]
            if pd.isna(team) or pd.isna(md):
                days_rest.append(np.nan); btb.append(0); bye.append(0)
                continue
            prev = team_prev_date.get((team, i), pd.NaT)
            if pd.notna(prev):
                rest = (md - prev).days
                days_rest.append(rest); btb.append(1 if rest < 6 else 0); bye.append(1 if rest >= 11 else 0)
            else:
                days_rest.append(np.nan); btb.append(0); bye.append(0)

        df[f"{side}_days_rest"] = days_rest
        df[f"{side}_is_back_to_back"] = btb
        df[f"{side}_bye_last_round"] = bye

    df["rest_diff"] = pd.to_numeric(df["home_days_rest"], errors="coerce") - pd.to_numeric(df["away_days_rest"], errors="coerce")
    print(f"  Added schedule features")
    return df


# =========================================================================
# STEP 11: Contextual features
# =========================================================================
def compute_contextual_features(matches):
    """Compute contextual features."""
    print("\n" + "=" * 80)
    print("  STEP 11: COMPUTING CONTEXTUAL FEATURES")
    print("=" * 80)

    df = matches.copy()
    df["is_home"] = 1
    df["round_number"] = pd.to_numeric(df["round"], errors="coerce")
    finals_keywords = {"final", "qualif", "elim", "semi", "prelim", "grand"}
    df["is_finals"] = df["round"].astype(str).str.lower().apply(
        lambda r: int(any(kw in r for kw in finals_keywords))
    )
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["season_year"] = df["season"]
    print(f"  Added contextual features")
    return df


# =========================================================================
# STEP 12: Engineered interaction features + momentum
# =========================================================================
def compute_engineered_features(df):
    """Compute interaction and derived features."""
    print("\n" + "=" * 80)
    print("  STEP 12: COMPUTING ENGINEERED & MOMENTUM FEATURES")
    print("=" * 80)

    df = df.copy()

    # Momentum: form_momentum_3v8 = win_rate_last_3 - win_rate_last_8
    df["home_form_momentum"] = df["home_win_rate_3"] - df["home_win_rate_8"]
    df["away_form_momentum"] = df["away_win_rate_3"] - df["away_win_rate_8"]
    df["form_momentum_diff"] = df["home_form_momentum"] - df["away_form_momentum"]

    # Momentum 3v5
    df["home_form_momentum_3v5"] = df["home_win_rate_3"] - df["home_win_rate_5"]
    df["away_form_momentum_3v5"] = df["away_win_rate_3"] - df["away_win_rate_5"]

    # Elo interactions
    df["elo_diff_sq"] = df["elo_diff"] ** 2 * np.sign(df["elo_diff"])
    df["elo_diff_abs"] = df["elo_diff"].abs()

    # Odds-Elo divergence (market disagrees with Elo)
    if "odds_home_prob" in df.columns:
        df["odds_elo_diff"] = df["odds_home_prob"] - df["home_elo_prob"]
        df["odds_elo_abs_diff"] = df["odds_elo_diff"].abs()

    # Attack-defense differential
    df["home_attack_defense_3"] = df["home_avg_pf_3"] - df["home_avg_pa_3"]
    df["away_attack_defense_3"] = df["away_avg_pf_3"] - df["away_avg_pa_3"]
    df["attack_defense_diff_3"] = df["home_attack_defense_3"] - df["away_attack_defense_3"]

    # Season progress
    df["season_progress"] = df["round_number"].clip(upper=26) / 26.0
    df["elo_diff_x_progress"] = df["elo_diff"] * df["season_progress"]

    # Competition points ratio
    total_cp = df["home_competition_points"] + df["away_competition_points"]
    df["comp_points_ratio"] = np.where(total_cp > 0, df["home_competition_points"] / total_cp, 0.5)

    # Composite strength
    df["home_strength"] = 0.4 * df["home_elo_prob"] + 0.3 * df["home_win_rate_5"].fillna(0.5) + 0.3 * (1 - df["home_ladder_pos"].fillna(9) / 17.0)
    df["away_strength"] = 0.4 * (1 - df["home_elo_prob"]) + 0.3 * df["away_win_rate_5"].fillna(0.5) + 0.3 * (1 - df["away_ladder_pos"].fillna(9) / 17.0)
    df["strength_diff"] = df["home_strength"] - df["away_strength"]

    # Cross interactions
    df["elo_x_rest"] = df["elo_diff"] * df["rest_diff"]
    df["ladder_x_finals"] = df["ladder_pos_diff"] * df["is_finals"].fillna(0)

    # Streak differential
    df["streak_diff"] = df["home_streak"] - df["away_streak"]

    # Halftime/penalty differentials
    df["halftime_lead_diff"] = df["home_avg_halftime_lead_5"] - df["away_avg_halftime_lead_5"]
    df["penalty_diff_diff"] = df["home_avg_penalty_diff_5"] - df["away_avg_penalty_diff_5"]

    # Home/away split differential (KEY feature)
    df["home_away_split_diff"] = df["home_team_home_win_pct"] - df["away_team_away_win_pct"]

    # Venue advantage differential
    df["venue_wr_diff"] = df["home_venue_win_rate"] - df["away_venue_win_rate"]

    print(f"  Added engineered features")
    return df


# =========================================================================
# STEP 13: Build target and final feature matrix
# =========================================================================
def build_feature_matrix(df):
    """Build the final feature matrix with all features."""
    print("\n" + "=" * 80)
    print("  STEP 13: BUILDING FINAL FEATURE MATRIX")
    print("=" * 80)

    df["home_win"] = np.where(
        df["home_score"] > df["away_score"], 1.0,
        np.where(df["home_score"] < df["away_score"], 0.0, np.nan)
    )

    # Define feature columns
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

    # Ladder home/away splits (NEW)
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

    # Odds (rich) (NEW)
    feature_cols += [
        "odds_home_prob", "odds_away_prob", "odds_home_favourite",
        "odds_home_open_prob", "odds_away_open_prob",
        "spread_home_open", "total_line_open",
        "odds_home_range", "odds_away_range",
        "bookmakers_surveyed",
        "odds_movement", "odds_movement_abs",
    ]

    # H2H (NEW)
    feature_cols += [
        "h2h_home_win_rate_3", "h2h_home_win_rate_5", "h2h_home_win_rate_all",
        "h2h_avg_margin_3", "h2h_avg_margin_5", "h2h_avg_margin_all",
        "h2h_matches_3", "h2h_matches_5", "h2h_matches_all",
    ]

    # Venue (NEW)
    feature_cols += [
        "home_venue_win_rate", "away_venue_win_rate",
        "venue_avg_total_score", "is_neutral_venue",
    ]

    # Momentum/Trend (NEW)
    feature_cols += [
        "home_form_momentum", "away_form_momentum", "form_momentum_diff",
        "home_form_momentum_3v5", "away_form_momentum_3v5",
        "home_streak", "away_streak", "streak_diff",
        "home_last_result", "away_last_result",
    ]

    # Halftime/Penalty (NEW)
    feature_cols += [
        "home_avg_halftime_lead_5", "away_avg_halftime_lead_5",
        "home_avg_penalty_diff_5", "away_avg_penalty_diff_5",
        "halftime_lead_diff", "penalty_diff_diff",
    ]

    # Engineered
    feature_cols += [
        "elo_diff_sq", "elo_diff_abs",
        "odds_elo_diff", "odds_elo_abs_diff",
        "home_attack_defense_3", "away_attack_defense_3", "attack_defense_diff_3",
        "season_progress", "elo_diff_x_progress",
        "comp_points_ratio", "home_strength", "away_strength", "strength_diff",
        "elo_x_rest", "ladder_x_finals",
        "home_away_split_diff", "venue_wr_diff",
    ]

    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    # De-duplicate
    seen = set()
    feature_cols_unique = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            feature_cols_unique.append(c)
    feature_cols = feature_cols_unique

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

    # Missing summary
    missing = features[feature_cols].isnull().sum()
    missing_pct = missing / len(features) * 100
    high_miss = missing_pct[missing_pct > 10].sort_values(ascending=False)
    if len(high_miss) > 0:
        print(f"\n  Features with >10% missing:")
        for c, pct in high_miss.items():
            print(f"    {c:<45} {pct:.1f}%")

    return features, feature_cols


# =========================================================================
# STEP 14: Walk-forward backtesting
# =========================================================================
def fill_missing(X_train, X_test):
    """Fill NaN using train medians; boolean flags with 0."""
    bool_cols = {"home_is_back_to_back", "away_is_back_to_back",
                 "home_bye_last_round", "away_bye_last_round",
                 "is_finals", "odds_home_favourite", "is_home", "is_neutral_venue"}
    medians = X_train.median()
    Xtr = X_train.copy()
    Xte = X_test.copy()
    for col in X_train.columns:
        if col in bool_cols:
            Xtr[col] = Xtr[col].fillna(0)
            Xte[col] = Xte[col].fillna(0)
        else:
            Xtr[col] = Xtr[col].fillna(medians.get(col, 0))
            Xte[col] = Xte[col].fillna(medians.get(col, 0))
    return Xtr, Xte


def compute_sample_weights(years, decay=0.9):
    """Exponential decay weighting: recent seasons get more weight."""
    max_yr = years.max()
    return decay ** (max_yr - years)


def select_top_features(X_train, y_train, feature_cols, sw, top_n=50):
    """Use XGBoost importance to select top N features."""
    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.02,
                               verbosity=0, random_state=42)
    model.fit(X_train, y_train, sample_weight=sw)
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return list(imp.head(top_n).index)


def walk_forward_backtest(features, feature_cols):
    """Run walk-forward backtesting with all models."""
    print("\n" + "=" * 80)
    print("  STEP 14: WALK-FORWARD BACKTESTING")
    print("=" * 80)

    df = features.copy()
    all_results = {}

    # Storage per model per fold
    model_names = ["XGBoost", "LightGBM", "CatBoost", "LogReg",
                   "XGB_top50", "LGB_top50", "CAT_top50",
                   "Odds Implied"]
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

        # Sample weights
        sw = compute_sample_weights(pd.Series(train_years), decay=0.9)

        y_parts.append(y_test)
        odds_parts.append(odds_test)
        year_parts.append(np.full(len(y_test), test_year))

        print(f"\n  Fold {fold_idx+1}: Train <=2013-{train_end} ({len(X_train)}) -> Test {test_year} ({len(X_test)})")

        # --- Feature selection for top-N variants ---
        top50 = select_top_features(X_train, y_train, feature_cols, sw, top_n=50)
        X_train_top = X_train[top50]
        X_test_top = X_test[top50]

        # --- Odds Implied ---
        model_oof["Odds Implied"].append(np.clip(odds_test, 1e-7, 1-1e-7))

        # --- XGBoost (all features) ---
        xgb_model = xgb.XGBClassifier(**BEST_XGB_PARAMS)
        xgb_model.fit(X_train, y_train, sample_weight=sw)
        xgb_prob = np.clip(xgb_model.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7)
        model_oof["XGBoost"].append(xgb_prob)

        # --- XGBoost (top 50) ---
        xgb_model2 = xgb.XGBClassifier(**BEST_XGB_PARAMS)
        xgb_model2.fit(X_train_top, y_train, sample_weight=sw)
        xgb_prob2 = np.clip(xgb_model2.predict_proba(X_test_top)[:, 1], 1e-7, 1-1e-7)
        model_oof["XGB_top50"].append(xgb_prob2)

        # --- LightGBM (all features) ---
        lgb_model = lgbm.LGBMClassifier(**BEST_LGB_PARAMS)
        lgb_model.fit(X_train, y_train, sample_weight=sw)
        lgb_prob = np.clip(lgb_model.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7)
        model_oof["LightGBM"].append(lgb_prob)

        # --- LightGBM (top 50) ---
        lgb_model2 = lgbm.LGBMClassifier(**BEST_LGB_PARAMS)
        lgb_model2.fit(X_train_top, y_train, sample_weight=sw)
        lgb_prob2 = np.clip(lgb_model2.predict_proba(X_test_top)[:, 1], 1e-7, 1-1e-7)
        model_oof["LGB_top50"].append(lgb_prob2)

        # --- CatBoost (all features) ---
        cat_model = CatBoostClassifier(**BEST_CAT_PARAMS)
        cat_model.fit(X_train, y_train, sample_weight=sw)
        cat_prob = np.clip(cat_model.predict_proba(X_test)[:, 1], 1e-7, 1-1e-7)
        model_oof["CatBoost"].append(cat_prob)

        # --- CatBoost (top 50) ---
        cat_model2 = CatBoostClassifier(**BEST_CAT_PARAMS)
        cat_model2.fit(X_train_top, y_train, sample_weight=sw)
        cat_prob2 = np.clip(cat_model2.predict_proba(X_test_top)[:, 1], 1e-7, 1-1e-7)
        model_oof["CAT_top50"].append(cat_prob2)

        # --- Logistic Regression ---
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr_model.fit(X_train_sc, y_train, sample_weight=sw)
        lr_prob = np.clip(lr_model.predict_proba(X_test_sc)[:, 1], 1e-7, 1-1e-7)
        model_oof["LogReg"].append(lr_prob)

        # Print fold results (just the key models)
        for n in ["XGBoost", "XGB_top50", "CatBoost", "CAT_top50", "LogReg", "Odds Implied"]:
            m = compute_metrics(y_test, model_oof[n][-1])
            print(f"    {n:15s}: Acc={m['accuracy']:.3f}  LL={m['log_loss']:.4f}")

    # --- Compute per-model aggregate metrics ---
    print("\n" + "-" * 80)
    print("  AGGREGATE RESULTS (avg across folds)")
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
            print(f"  {n:15s}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}  Brier={result['brier']:.4f}  AUC={result['auc']:.4f}")

    return all_results, model_oof, y_parts, odds_parts, year_parts


# =========================================================================
# STEP 15: Odds-blended models + stacking
# =========================================================================
def blend_and_stack(all_results, model_oof, y_parts, odds_parts):
    """Compute odds-blended models and stacking ensemble."""
    print("\n" + "=" * 80)
    print("  STEP 15: ODDS-BLEND & STACKING")
    print("=" * 80)

    ml_models = ["XGBoost", "LightGBM", "CatBoost", "LogReg",
                  "XGB_top50", "LGB_top50", "CAT_top50"]
    active_folds = [i for i in range(len(FOLDS)) if len(y_parts[i]) > 0]

    # --- Odds-blend per model (fine-grained weight search) ---
    print("\n  Odds-Blended Models:")
    for name in ml_models:
        best_w, best_ll = None, float("inf")
        for w_int in range(5, 500, 5):
            w = w_int / 1000.0
            fold_lls = []
            for i in active_folds:
                blended = w * model_oof[name][i] + (1 - w) * odds_parts[i]
                fold_lls.append(safe_log_loss(y_parts[i], blended))
            avg_ll = np.mean(fold_lls)
            if avg_ll < best_ll:
                best_ll = avg_ll
                best_w = w

        # Compute metrics at best weight
        fold_metrics = []
        for i in active_folds:
            blended = best_w * model_oof[name][i] + (1 - best_w) * odds_parts[i]
            fold_metrics.append(compute_metrics(y_parts[i], blended))

        result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        label = f"Odds-Blend {name} (w={best_w:.3f})"
        all_results[label] = result
        print(f"    {label}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    # --- Walk-forward blend (per-fold weight optimization using prior folds) ---
    print("\n  Walk-Forward Odds-Blended Models:")
    for name in ml_models:
        blend_probs_per_fold = []
        for fold_idx in range(len(FOLDS)):
            if len(y_parts[fold_idx]) == 0:
                blend_probs_per_fold.append(np.array([]))
                continue
            if fold_idx < 2:
                best_w = 0.15
            else:
                best_w_wf, best_ll_wf = None, float("inf")
                for w_int in range(0, 505, 5):
                    w = w_int / 1000.0
                    prior_lls = []
                    for prev in range(fold_idx):
                        if len(y_parts[prev]) == 0:
                            continue
                        blended = w * model_oof[name][prev] + (1 - w) * odds_parts[prev]
                        prior_lls.append(safe_log_loss(y_parts[prev], blended))
                    if prior_lls:
                        avg_ll = np.mean(prior_lls)
                        if avg_ll < best_ll_wf:
                            best_ll_wf = avg_ll
                            best_w_wf = w
                best_w = best_w_wf if best_w_wf is not None else 0.15

            blended = best_w * model_oof[name][fold_idx] + (1 - best_w) * odds_parts[fold_idx]
            blend_probs_per_fold.append(blended)

        fold_metrics = []
        for i in active_folds:
            fold_metrics.append(compute_metrics(y_parts[i], blend_probs_per_fold[i]))
        result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        label = f"WF-Blend {name}"
        all_results[label] = result
        print(f"    {label}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    # --- Multi-model blend (GBM average + odds) ---
    print("\n  Multi-Model Blends:")

    # Several GBM averages
    for avg_label, avg_names in [
        ("GBM_Avg", ["XGBoost", "LightGBM", "CatBoost"]),
        ("GBM_Top50_Avg", ["XGB_top50", "LGB_top50", "CAT_top50"]),
        ("GBM_Mixed_Avg", ["XGB_top50", "LightGBM", "CatBoost"]),
    ]:
        avg_probs = []
        for i in range(len(FOLDS)):
            if len(y_parts[i]) == 0:
                avg_probs.append(np.array([]))
            else:
                avg_probs.append(np.mean([model_oof[n][i] for n in avg_names], axis=0))

        best_w, best_ll = None, float("inf")
        for w_int in range(5, 500, 5):
            w = w_int / 1000.0
            fold_lls = []
            for i in active_folds:
                blended = w * avg_probs[i] + (1 - w) * odds_parts[i]
                fold_lls.append(safe_log_loss(y_parts[i], blended))
            avg_ll = np.mean(fold_lls)
            if avg_ll < best_ll:
                best_ll = avg_ll
                best_w = w

        fold_metrics = []
        for i in active_folds:
            blended = best_w * avg_probs[i] + (1 - best_w) * odds_parts[i]
            fold_metrics.append(compute_metrics(y_parts[i], blended))
        result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        label = f"Odds-Blend {avg_label} (w={best_w:.3f})"
        all_results[label] = result
        print(f"    {label}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    # --- Scipy multi-model weighted blend ---
    def multi_blend_objective(weights, model_names_list, oof_dict, odds_p, y_p, active_f):
        w_odds = 1.0 - np.sum(weights)
        if w_odds < 0:
            return 10.0
        fold_lls = []
        for i in active_f:
            blended = np.zeros_like(y_p[i], dtype=float)
            for j, n in enumerate(model_names_list):
                blended += weights[j] * oof_dict[n][i]
            blended += w_odds * odds_p[i]
            fold_lls.append(safe_log_loss(y_p[i], blended))
        return np.mean(fold_lls)

    for combo_name, combo in [
        ("XGB+CAT+Odds", ["XGBoost", "CatBoost"]),
        ("3GBM+Odds", ["XGBoost", "LightGBM", "CatBoost"]),
        ("All+Odds", ["XGBoost", "LightGBM", "CatBoost", "LogReg"]),
        ("XGB50+CAT50+Odds", ["XGB_top50", "CAT_top50"]),
        ("3GBM50+Odds", ["XGB_top50", "LGB_top50", "CAT_top50"]),
        ("XGB+XGB50+CAT+Odds", ["XGBoost", "XGB_top50", "CatBoost"]),
        ("6GBM+Odds", ["XGBoost", "LightGBM", "CatBoost", "XGB_top50", "LGB_top50", "CAT_top50"]),
        ("AllModels+Odds", ["XGBoost", "LightGBM", "CatBoost", "LogReg", "XGB_top50", "LGB_top50", "CAT_top50"]),
    ]:
        n_m = len(combo)
        x0 = np.array([0.1 / n_m] * n_m)
        res = minimize(
            multi_blend_objective, x0, args=(combo, model_oof, odds_parts, y_parts, active_folds),
            method="Nelder-Mead", options={"maxiter": 5000, "xatol": 0.001, "fatol": 1e-7}
        )
        bw = res.x
        w_odds = 1.0 - np.sum(bw)

        fold_metrics = []
        for i in active_folds:
            blended = np.zeros_like(y_parts[i], dtype=float)
            for j, n in enumerate(combo):
                blended += bw[j] * model_oof[n][i]
            blended += w_odds * odds_parts[i]
            fold_metrics.append(compute_metrics(y_parts[i], blended))

        result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        weight_str = ", ".join([f"{n}={bw[j]:.3f}" for j, n in enumerate(combo)]) + f", odds={w_odds:.3f}"
        label = f"OptBlend {combo_name} ({weight_str})"
        all_results[label] = result
        print(f"    {label}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    # --- Stacking Ensembles ---
    print("\n  Stacking Ensembles:")
    for stack_label, stack_models in [
        ("3GBM+Odds", ["XGBoost", "LightGBM", "CatBoost"]),
        ("3GBM50+Odds", ["XGB_top50", "LGB_top50", "CAT_top50"]),
        ("6GBM+Odds", ["XGBoost", "LightGBM", "CatBoost", "XGB_top50", "LGB_top50", "CAT_top50"]),
        ("All+Odds", ["XGBoost", "LightGBM", "CatBoost", "LogReg", "XGB_top50", "LGB_top50", "CAT_top50"]),
    ]:
        stack_probs = []
        for fold_idx in range(len(FOLDS)):
            if len(y_parts[fold_idx]) == 0:
                stack_probs.append(np.array([]))
                continue

            meta_X_train_parts = []
            meta_y_train_parts = []
            for prev in range(fold_idx):
                if len(y_parts[prev]) == 0:
                    continue
                row = np.column_stack([model_oof[n][prev] for n in stack_models] + [odds_parts[prev]])
                meta_X_train_parts.append(row)
                meta_y_train_parts.append(y_parts[prev])

            if len(meta_X_train_parts) < 1:
                avg = np.mean([model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]], axis=0)
                stack_probs.append(np.clip(avg, 1e-7, 1-1e-7))
                continue

            meta_X_train = np.vstack(meta_X_train_parts)
            meta_y_train = np.concatenate(meta_y_train_parts)
            meta_X_test = np.column_stack([model_oof[n][fold_idx] for n in stack_models] + [odds_parts[fold_idx]])

            meta_lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            meta_lr.fit(meta_X_train, meta_y_train)
            meta_prob = np.clip(meta_lr.predict_proba(meta_X_test)[:, 1], 1e-7, 1-1e-7)
            stack_probs.append(meta_prob)

        fold_metrics = []
        for i in active_folds:
            fold_metrics.append(compute_metrics(y_parts[i], stack_probs[i]))
        result = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        label = f"Stacking ({stack_label} -> LR)"
        all_results[label] = result
        print(f"    {label}: Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}")

    return all_results


# =========================================================================
# STEP 16: Comparison with v2
# =========================================================================
def print_comparison(all_results):
    """Print comprehensive comparison."""
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE RESULTS COMPARISON (V3 Features)")
    print("=" * 80)

    # v2 baselines for reference
    v2_reference = {
        "Odds-Blend XGB (v2)": {"accuracy": 0.6790, "log_loss": 0.5995},
        "Odds Implied (v2)": {"accuracy": 0.6800, "log_loss": 0.5999},
    }

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

    print()
    hdr = f"{'#':>3}  {'Model':<60} | {'Acc':>7} | {'LL':>7} | {'Brier':>7} | {'AUC':>7} | {'vs Odds':>8} | {'vs v2 best':>10}"
    print(hdr)
    print("-" * len(hdr))

    for idx, row in comp_df.iterrows():
        ll_diff = row["Log Loss"] - odds_ll
        v2_diff = row["Log Loss"] - 0.5995  # v2 best
        marker = ""
        if row["Model"] == "Odds Implied":
            marker = " <-BASE"
        elif row["Log Loss"] < odds_ll:
            marker = " ***"

        print(
            f"{idx+1:3d}  {row['Model']:<60} | {row['Accuracy']:7.4f} | "
            f"{row['Log Loss']:7.4f} | {row['Brier']:7.4f} | {row['AUC']:7.4f} | "
            f"{ll_diff:+8.4f} | {v2_diff:+10.4f}{marker}"
        )

    print("-" * len(hdr))

    # Summary
    best = comp_df.iloc[0]
    print(f"\n  BEST MODEL: {best['Model']}")
    print(f"    Accuracy:  {best['Accuracy']:.4f}  (v2 best: 0.6790, odds: {odds_acc:.4f})")
    print(f"    Log Loss:  {best['Log Loss']:.4f}  (v2 best: 0.5995, odds: {odds_ll:.4f})")

    if best["Log Loss"] < odds_ll:
        imp = (odds_ll - best["Log Loss"]) / odds_ll * 100
        print(f"\n    >>> BEATS ODDS BASELINE by {imp:.3f}% in log loss <<<")

    if best["Log Loss"] < 0.5995:
        imp_v2 = (0.5995 - best["Log Loss"]) / 0.5995 * 100
        print(f"    >>> BEATS V2 BEST by {imp_v2:.3f}% in log loss <<<")

    beats_odds = comp_df[comp_df["Log Loss"] < odds_ll]
    print(f"\n    {len(beats_odds)} out of {len(comp_df)} models beat the odds baseline in log loss.")

    beats_v2 = comp_df[comp_df["Log Loss"] < 0.5995]
    print(f"    {len(beats_v2)} out of {len(comp_df)} models beat the v2 best in log loss.")

    return comp_df


# =========================================================================
# MAIN
# =========================================================================
def main():
    overall_start = time.time()

    print()
    print("*" * 80)
    print("*  NRL MATCH PREDICTION - ENHANCED FEATURE ENGINEERING & RETRAINING (V3)")
    print("*  Goal: Beat v2 best (67.9% acc / 0.5995 LL) and odds baseline (68.0% / 0.5999)")
    print("*" * 80)
    print()

    # Step 1: Load data
    matches, ladders, odds = load_and_fix_homeaway()

    # Step 2: Link odds
    matches = link_odds(matches, odds)

    # Step 3: Tune Elo
    elo_params = tune_elo(matches, n_trials=50)

    # Step 4: Backfill Elo with optimal params
    matches = backfill_elo(matches, elo_params)

    # Step 5: Rolling form features (incl. halftime lead, penalty diff, streak)
    matches = compute_rolling_form_features(matches)

    # Step 6: H2H features
    matches = compute_h2h_features(matches)

    # Step 7: Ladder features (incl. home/away splits)
    matches = compute_ladder_features(matches, ladders)

    # Step 8: Venue features
    matches = compute_venue_features(matches)

    # Step 9: Odds features (rich)
    matches = compute_odds_features(matches)

    # Step 10: Schedule features
    matches = compute_schedule_features(matches)

    # Step 11: Contextual features
    matches = compute_contextual_features(matches)

    # Step 12: Engineered features
    matches = compute_engineered_features(matches)

    # Step 13: Build feature matrix
    features, feature_cols = build_feature_matrix(matches)

    # Save v3 features
    output_path = FEATURES_DIR / "features_v3.parquet"
    features.to_parquet(output_path, index=False)
    print(f"\n  Saved features_v3.parquet: {output_path}")
    print(f"  Shape: {features.shape}")

    # Step 14: Walk-forward backtesting
    all_results, model_oof, y_parts, odds_parts, year_parts = walk_forward_backtest(features, feature_cols)

    # Step 15: Odds-blend and stacking
    all_results = blend_and_stack(all_results, model_oof, y_parts, odds_parts)

    # Step 16: Print comparison
    comp_df = print_comparison(all_results)

    # Save comparison
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(report_dir / "v3_results_comparison.csv", index=False)

    elapsed = time.time() - overall_start
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\n" + "=" * 80)
    print("  V3 PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
