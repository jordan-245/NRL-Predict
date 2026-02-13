"""
Honest Evaluation Experiment
=============================
Two focused tests to determine if our models add real information beyond odds:

1. CONSTRAINED BLENDING: Simple averages and non-negative-weight blends of
   3-4 models + odds, evaluated strictly via walk-forward (no look-ahead).

2. FEATURE PRUNING: Train individual models on top 30/50 features by SHAP
   importance. If pruned models can beat odds LL=0.5999, features have signal.

If neither approach beats odds in walk-forward, the honest conclusion is that
the current feature set doesn't contain information the market hasn't priced in.

Usage:
    python experiment_honest_eval.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_DIR = PROJECT_ROOT / "data" / "features"

import run_v4_pipeline as v4

# === FOLDS ===
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

# === HYPERPARAMETERS ===
BEST_XGB_PARAMS = v4.BEST_XGB_PARAMS.copy()
BEST_LGB_PARAMS = v4.BEST_LGB_PARAMS.copy()
BEST_CAT_PARAMS = v4.BEST_CAT_PARAMS.copy()
BEST_RF_PARAMS = {
    'n_estimators': 234, 'max_depth': 10, 'min_samples_leaf': 23,
    'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1,
}
SAMPLE_WEIGHT_DECAY = 0.920

ODDS_FEATURE_SUBSTRINGS = [
    "odds", "spread", "total_line", "h2h_", "implied_draw",
    "draw_competitiveness", "market_confidence", "bookmakers",
    "overround", "odds_movement", "fav_consistency", "elo_spread_agree",
    "odds_spread", "scoring_env_ratio",
]


def safe_log_loss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return log_loss(y, p)


def compute_metrics(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": safe_log_loss(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
    }


def fill_missing(X_train, X_test):
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
        fill_val = 0 if col in bool_cols else (medians.get(col, 0) if pd.notna(medians.get(col, 0)) else 0)
        Xtr[col] = Xtr[col].fillna(fill_val)
        Xte[col] = Xte[col].fillna(fill_val)
    return Xtr, Xte


def compute_sample_weights(years, decay=SAMPLE_WEIGHT_DECAY):
    max_yr = years.max()
    return decay ** (max_yr - years)


def get_shap_top_features(X_train, y_train, sw, feature_cols, top_n=50):
    """Get top N features by XGBoost gain importance (fast SHAP proxy)."""
    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.02,
                               verbosity=0, random_state=42)
    model.fit(X_train, y_train, sample_weight=sw)
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return list(imp.head(top_n).index)


def train_model(name, params, X_train, y_train, sw):
    """Train a model by name and return the fitted model."""
    if name == "XGBoost":
        m = xgb.XGBClassifier(**params["xgb"])
        m.fit(X_train, y_train, sample_weight=sw)
    elif name == "LightGBM":
        m = lgbm.LGBMClassifier(**params["lgb"])
        m.fit(X_train, y_train, sample_weight=sw)
    elif name == "CatBoost":
        m = CatBoostClassifier(**params["cat"])
        m.fit(X_train, y_train, sample_weight=sw)
    elif name == "RandomForest":
        m = RandomForestClassifier(**params["rf"])
        m.fit(X_train, y_train, sample_weight=sw)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


# ==========================================================================
#  EXPERIMENT 1: CONSTRAINED WALK-FORWARD BLENDING
# ==========================================================================

def run_constrained_blending(features, feature_cols):
    """
    Walk-forward blending with honest constraints:
    - Simple equal-weight averages
    - Non-negative weights summing to 1
    - Small model sets (3-4 models + odds)
    """
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: CONSTRAINED WALK-FORWARD BLENDING")
    print("=" * 80)

    df = features.copy()
    n_folds = len(FOLDS)

    core_models = ["XGBoost", "CatBoost", "RandomForest", "LightGBM"]
    params = {
        "xgb": BEST_XGB_PARAMS,
        "lgb": BEST_LGB_PARAMS,
        "cat": BEST_CAT_PARAMS,
        "rf": BEST_RF_PARAMS,
    }

    # Collect OOF predictions per fold
    model_oof = {n: [None] * n_folds for n in core_models}
    odds_parts = [None] * n_folds
    y_parts = [None] * n_folds
    year_parts = [None] * n_folds

    for fold_idx, fold in enumerate(FOLDS):
        train_end = fold["train_end"]
        test_year = fold["test_year"]
        train_mask = df["year"] <= train_end
        test_mask = df["year"] == test_year

        if test_mask.sum() == 0:
            y_parts[fold_idx] = np.array([])
            odds_parts[fold_idx] = np.array([])
            for n in core_models:
                model_oof[n][fold_idx] = np.array([])
            continue

        X_train_raw = df.loc[train_mask, feature_cols].copy()
        X_test_raw = df.loc[test_mask, feature_cols].copy()
        y_train = df.loc[train_mask, "home_win"].values
        y_test = df.loc[test_mask, "home_win"].values
        train_years = df.loc[train_mask, "year"].values

        odds_test = df.loc[test_mask, "odds_home_prob"].values
        odds_test = np.where(np.isnan(odds_test), 0.55, odds_test)
        odds_parts[fold_idx] = odds_test
        y_parts[fold_idx] = y_test
        year_parts[fold_idx] = np.full(len(y_test), test_year)

        X_train, X_test = fill_missing(X_train_raw, X_test_raw)
        sw = compute_sample_weights(pd.Series(train_years))

        print(f"\n  Fold {fold_idx+1}: Train <={train_end} ({len(X_train)}) "
              f"-> Test {test_year} ({len(X_test)})")

        for name in core_models:
            m = train_model(name, params, X_train, y_train, sw)
            preds = np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1 - 1e-7)
            model_oof[name][fold_idx] = preds
            ll = safe_log_loss(y_test, preds)
            print(f"    {name:15s}: LL={ll:.4f}")

        odds_ll = safe_log_loss(y_test, odds_test)
        print(f"    {'Odds':15s}: LL={odds_ll:.4f}")

    active_folds = [i for i in range(n_folds) if len(y_parts[i]) > 0]

    # --- Individual model aggregate results ---
    print("\n" + "-" * 70)
    print("  INDIVIDUAL MODEL RESULTS (walk-forward)")
    print("-" * 70)
    for name in core_models + ["Odds"]:
        fold_metrics = []
        for i in active_folds:
            if name == "Odds":
                fold_metrics.append(compute_metrics(y_parts[i], odds_parts[i]))
            else:
                fold_metrics.append(compute_metrics(y_parts[i], model_oof[name][i]))
        agg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        marker = " <-- BASELINE" if name == "Odds" else ""
        print(f"  {name:15s}: Acc={agg['accuracy']:.4f}  LL={agg['log_loss']:.4f}"
              f"  Brier={agg['brier']:.4f}{marker}")

    # --- Blending strategies ---
    print("\n" + "-" * 70)
    print("  WALK-FORWARD BLEND RESULTS (all honestly evaluated)")
    print("-" * 70)

    all_blend_results = {}

    # Strategy A: Simple equal-weight averages (no optimization at all)
    avg_combos = [
        ("EqualAvg-3GBM+Odds", ["XGBoost", "LightGBM", "CatBoost"]),
        ("EqualAvg-4Models+Odds", core_models),
        ("EqualAvg-XGB+CAT+Odds", ["XGBoost", "CatBoost"]),
        ("EqualAvg-XGB+RF+Odds", ["XGBoost", "RandomForest"]),
    ]

    print("\n  A) Simple equal-weight averages (no optimizer):")
    for combo_name, combo in avg_combos:
        fold_metrics = []
        for i in active_folds:
            n_m = len(combo)
            # Equal weight: each model gets w, odds gets w too
            w = 1.0 / (n_m + 1)
            blended = np.zeros_like(y_parts[i], dtype=float)
            for name in combo:
                blended += w * model_oof[name][i]
            blended += w * odds_parts[i]
            blended = np.clip(blended, 1e-7, 1 - 1e-7)
            fold_metrics.append(compute_metrics(y_parts[i], blended))
        agg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        all_blend_results[combo_name] = agg
        print(f"    {combo_name:35s}: LL={agg['log_loss']:.4f}  Acc={agg['accuracy']:.4f}")

    # Strategy B: Odds-heavy equal mixes (give odds more weight)
    print("\n  B) Odds-heavy mixes (models share 20-40%, odds gets rest):")
    for model_share in [0.20, 0.30, 0.40]:
        for combo_name_base, combo in [("3GBM", ["XGBoost", "LightGBM", "CatBoost"]),
                                        ("4Models", core_models)]:
            label = f"OddsHeavy-{combo_name_base}-{int(model_share*100)}pct"
            fold_metrics = []
            for i in active_folds:
                n_m = len(combo)
                w_model = model_share / n_m
                w_odds = 1.0 - model_share
                blended = np.zeros_like(y_parts[i], dtype=float)
                for name in combo:
                    blended += w_model * model_oof[name][i]
                blended += w_odds * odds_parts[i]
                blended = np.clip(blended, 1e-7, 1 - 1e-7)
                fold_metrics.append(compute_metrics(y_parts[i], blended))
            agg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            all_blend_results[label] = agg
            print(f"    {label:35s}: LL={agg['log_loss']:.4f}  Acc={agg['accuracy']:.4f}")

    # Strategy C: Walk-forward with NON-NEGATIVE weights summing to 1
    print("\n  C) Walk-forward OptBlend with NON-NEGATIVE weights (sum=1):")

    def constrained_blend_obj(weights, names_list, oof_dict, odds_p, y_p, folds_list):
        """Objective: weights for models, last weight element is for odds. All non-neg, sum=1."""
        total = np.sum(weights)
        if abs(total - 1.0) > 0.01:
            return 10.0
        fold_lls = []
        for i in folds_list:
            blended = np.zeros_like(y_p[i], dtype=float)
            for j, n in enumerate(names_list):
                blended += weights[j] * oof_dict[n][i]
            blended += weights[len(names_list)] * odds_p[i]  # odds weight is last
            blended = np.clip(blended, 1e-7, 1 - 1e-7)
            fold_lls.append(safe_log_loss(y_p[i], blended))
        return np.mean(fold_lls)

    wf_nn_combos = [
        ("WF-NN-3GBM+Odds", ["XGBoost", "LightGBM", "CatBoost"]),
        ("WF-NN-4Models+Odds", core_models),
        ("WF-NN-XGB+CAT+Odds", ["XGBoost", "CatBoost"]),
        ("WF-NN-XGB+RF+Odds", ["XGBoost", "RandomForest"]),
    ]

    for combo_name, combo in wf_nn_combos:
        wf_probs = [np.array([])] * n_folds
        for fold_idx in range(n_folds):
            if len(y_parts[fold_idx]) == 0:
                continue

            n_m = len(combo)
            if fold_idx < 2:
                # Not enough prior folds: use equal weights
                w = 1.0 / (n_m + 1)
                blended = np.zeros_like(y_parts[fold_idx], dtype=float)
                for name in combo:
                    blended += w * model_oof[name][fold_idx]
                blended += w * odds_parts[fold_idx]
            else:
                prior_folds = [p for p in range(fold_idx) if len(y_parts[p]) > 0]
                # n_m model weights + 1 odds weight, all non-negative, sum=1
                n_w = n_m + 1
                x0 = np.array([1.0 / n_w] * n_w)
                bounds = [(0.0, 1.0)] * n_w  # NON-NEGATIVE
                constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

                try:
                    res = minimize(
                        constrained_blend_obj, x0,
                        args=(combo, model_oof, odds_parts, y_parts, prior_folds),
                        method="SLSQP",
                        bounds=bounds,
                        constraints=constraints,
                        options={"maxiter": 2000, "ftol": 1e-9}
                    )
                    bw = res.x
                except Exception:
                    bw = x0  # fallback to equal

                blended = np.zeros_like(y_parts[fold_idx], dtype=float)
                for j, name in enumerate(combo):
                    blended += bw[j] * model_oof[name][fold_idx]
                blended += bw[n_m] * odds_parts[fold_idx]

            wf_probs[fold_idx] = np.clip(blended, 1e-7, 1 - 1e-7)

        fold_metrics = []
        for i in active_folds:
            if len(wf_probs[i]) == 0:
                continue
            fold_metrics.append(compute_metrics(y_parts[i], wf_probs[i]))
        if fold_metrics:
            agg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            all_blend_results[combo_name] = agg
            print(f"    {combo_name:35s}: LL={agg['log_loss']:.4f}  Acc={agg['accuracy']:.4f}")

    # Strategy D: Walk-forward UNCONSTRAINED (old approach for reference)
    print("\n  D) Walk-forward OptBlend UNCONSTRAINED (old approach, for reference):")
    def old_blend_obj(weights, names_list, oof_dict, odds_p, y_p, folds_list):
        w_odds = 1.0 - np.sum(weights)
        if w_odds < -0.5:
            return 10.0
        fold_lls = []
        for i in folds_list:
            blended = np.zeros_like(y_p[i], dtype=float)
            for j, n in enumerate(names_list):
                blended += weights[j] * oof_dict[n][i]
            blended += w_odds * odds_p[i]
            blended = np.clip(blended, 1e-7, 1 - 1e-7)
            fold_lls.append(safe_log_loss(y_p[i], blended))
        return np.mean(fold_lls)

    wf_old_combos = [
        ("WF-Old-3GBM+Odds", ["XGBoost", "LightGBM", "CatBoost"]),
        ("WF-Old-4Models+Odds", core_models),
    ]

    for combo_name, combo in wf_old_combos:
        wf_probs = [np.array([])] * n_folds
        for fold_idx in range(n_folds):
            if len(y_parts[fold_idx]) == 0:
                continue

            n_m = len(combo)
            if fold_idx < 2:
                equal_w = 0.15 / n_m
                blended = np.zeros_like(y_parts[fold_idx], dtype=float)
                for name in combo:
                    blended += equal_w * model_oof[name][fold_idx]
                blended += (1 - 0.15) * odds_parts[fold_idx]
            else:
                prior_folds = [p for p in range(fold_idx) if len(y_parts[p]) > 0]
                x0 = np.array([0.05 / n_m] * n_m)
                bounds = [(-1.0, 1.0)] * n_m
                try:
                    res = minimize(
                        old_blend_obj, x0,
                        args=(combo, model_oof, odds_parts, y_parts, prior_folds),
                        method="SLSQP",
                        bounds=bounds,
                        options={"maxiter": 2000, "ftol": 1e-9}
                    )
                except Exception:
                    res = minimize(
                        old_blend_obj, x0,
                        args=(combo, model_oof, odds_parts, y_parts, prior_folds),
                        method="Nelder-Mead",
                        options={"maxiter": 5000}
                    )
                bw = res.x
                w_odds = 1.0 - np.sum(bw)
                blended = np.zeros_like(y_parts[fold_idx], dtype=float)
                for j, name in enumerate(combo):
                    blended += bw[j] * model_oof[name][fold_idx]
                blended += w_odds * odds_parts[fold_idx]

            wf_probs[fold_idx] = np.clip(blended, 1e-7, 1 - 1e-7)

        fold_metrics = []
        for i in active_folds:
            if len(wf_probs[i]) == 0:
                continue
            fold_metrics.append(compute_metrics(y_parts[i], wf_probs[i]))
        if fold_metrics:
            agg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            all_blend_results[combo_name] = agg
            print(f"    {combo_name:35s}: LL={agg['log_loss']:.4f}  Acc={agg['accuracy']:.4f}")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1 SUMMARY: All methods vs Odds baseline (LL=0.5999)")
    print("=" * 70)

    # Add odds and individual models
    for name in core_models:
        fold_metrics = []
        for i in active_folds:
            fold_metrics.append(compute_metrics(y_parts[i], model_oof[name][i]))
        all_blend_results[f"Individual-{name}"] = {
            k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]
        }
    fold_metrics = []
    for i in active_folds:
        fold_metrics.append(compute_metrics(y_parts[i], odds_parts[i]))
    all_blend_results["Odds-Baseline"] = {
        k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]
    }

    sorted_results = sorted(all_blend_results.items(), key=lambda x: x[1]["log_loss"])
    odds_ll = all_blend_results["Odds-Baseline"]["log_loss"]

    for rank, (label, metrics) in enumerate(sorted_results, 1):
        diff = metrics["log_loss"] - odds_ll
        marker = " <-- ODDS" if label == "Odds-Baseline" else ""
        beat = " BEATS ODDS" if diff < -0.0001 else ""
        print(f"  {rank:2d}. {label:40s}  LL={metrics['log_loss']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}  vs_odds={diff:+.4f}{marker}{beat}")

    return model_oof, odds_parts, y_parts, active_folds


# ==========================================================================
#  EXPERIMENT 2: FEATURE PRUNING
# ==========================================================================

def run_feature_pruning(features, feature_cols):
    """
    Train individual models on top-30 and top-50 features by importance.
    Check if pruned models can beat odds LL=0.5999.
    """
    print("\n\n" + "=" * 80)
    print("  EXPERIMENT 2: FEATURE PRUNING (SHAP/Importance-based)")
    print("=" * 80)

    df = features.copy()
    n_folds = len(FOLDS)

    models_to_test = ["XGBoost", "CatBoost", "RandomForest", "LightGBM"]
    params = {
        "xgb": BEST_XGB_PARAMS,
        "lgb": BEST_LGB_PARAMS,
        "cat": BEST_CAT_PARAMS,
        "rf": BEST_RF_PARAMS,
    }

    feature_sets = {
        "All292": feature_cols,
        "Top50": None,   # computed per fold
        "Top30": None,   # computed per fold
    }

    # Track results: model -> feature_set -> fold_metrics
    results = {}

    for fs_name in feature_sets:
        print(f"\n  --- Feature set: {fs_name} ---")

        for model_name in models_to_test:
            key = f"{model_name}-{fs_name}"
            fold_metrics_list = []

            for fold_idx, fold in enumerate(FOLDS):
                train_end = fold["train_end"]
                test_year = fold["test_year"]
                train_mask = df["year"] <= train_end
                test_mask = df["year"] == test_year

                if test_mask.sum() == 0:
                    continue

                X_train_raw = df.loc[train_mask, feature_cols].copy()
                X_test_raw = df.loc[test_mask, feature_cols].copy()
                y_train = df.loc[train_mask, "home_win"].values
                y_test = df.loc[test_mask, "home_win"].values
                train_years = df.loc[train_mask, "year"].values
                sw = compute_sample_weights(pd.Series(train_years))

                X_train_filled, X_test_filled = fill_missing(X_train_raw, X_test_raw)

                if fs_name == "All292":
                    cols = feature_cols
                else:
                    top_n = 50 if fs_name == "Top50" else 30
                    cols = get_shap_top_features(X_train_filled, y_train, sw,
                                                  feature_cols, top_n)

                X_tr = X_train_filled[cols]
                X_te = X_test_filled[cols]

                m = train_model(model_name, params, X_tr, y_train, sw)
                preds = np.clip(m.predict_proba(X_te)[:, 1], 1e-7, 1 - 1e-7)
                fold_metrics_list.append(compute_metrics(y_test, preds))

                if fold_idx == 0 and fs_name != "All292":
                    print(f"    {model_name} {fs_name} sample features: "
                          f"{cols[:5]}...")

            agg = {k: np.mean([m[k] for m in fold_metrics_list])
                   for k in fold_metrics_list[0]}
            results[key] = agg

    # Also compute odds baseline
    fold_metrics_list = []
    for fold in FOLDS:
        test_mask = df["year"] == fold["test_year"]
        if test_mask.sum() == 0:
            continue
        y_test = df.loc[test_mask, "home_win"].values
        odds_test = df.loc[test_mask, "odds_home_prob"].values
        odds_test = np.where(np.isnan(odds_test), 0.55, odds_test)
        fold_metrics_list.append(compute_metrics(y_test, odds_test))
    odds_agg = {k: np.mean([m[k] for m in fold_metrics_list]) for k in fold_metrics_list[0]}
    odds_ll = odds_agg["log_loss"]

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2 SUMMARY: Feature pruning vs Odds (LL=0.5999)")
    print("=" * 70)
    print(f"\n  {'Model':15s} {'All292':>10s} {'Top50':>10s} {'Top30':>10s}  "
          f"{'Best':>10s} {'vs Odds':>10s}")
    print("  " + "-" * 70)

    for model_name in models_to_test:
        lls = {}
        for fs_name in ["All292", "Top50", "Top30"]:
            key = f"{model_name}-{fs_name}"
            lls[fs_name] = results[key]["log_loss"]
        best_fs = min(lls, key=lls.get)
        best_ll = lls[best_fs]
        diff = best_ll - odds_ll
        beat = " BEATS" if diff < -0.0001 else ""
        print(f"  {model_name:15s} {lls['All292']:10.4f} {lls['Top50']:10.4f} "
              f"{lls['Top30']:10.4f}  {best_fs:>10s} {diff:+10.4f}{beat}")

    print(f"  {'Odds Baseline':15s} {odds_ll:10.4f} {'':>10s} {'':>10s}  "
          f"{'':>10s} {0.0:+10.4f}")

    # Also show accuracy
    print(f"\n  {'Model':15s} {'All292':>10s} {'Top50':>10s} {'Top30':>10s}  (Accuracy)")
    print("  " + "-" * 55)
    for model_name in models_to_test:
        accs = {}
        for fs_name in ["All292", "Top50", "Top30"]:
            key = f"{model_name}-{fs_name}"
            accs[fs_name] = results[key]["accuracy"]
        print(f"  {model_name:15s} {accs['All292']:10.4f} {accs['Top50']:10.4f} "
              f"{accs['Top30']:10.4f}")
    print(f"  {'Odds':15s} {odds_agg['accuracy']:10.4f}")

    # Constrained blend with pruned features
    print("\n" + "-" * 70)
    print("  BONUS: Simple blend of pruned models + odds")
    print("-" * 70)

    for top_n, fs_label in [(50, "Top50"), (30, "Top30")]:
        # Retrain and collect OOF for blending
        pruned_oof = {n: [] for n in models_to_test}
        pruned_odds = []
        pruned_y = []

        for fold_idx, fold in enumerate(FOLDS):
            train_end = fold["train_end"]
            test_year = fold["test_year"]
            train_mask = df["year"] <= train_end
            test_mask = df["year"] == test_year

            if test_mask.sum() == 0:
                for n in models_to_test:
                    pruned_oof[n].append(np.array([]))
                pruned_odds.append(np.array([]))
                pruned_y.append(np.array([]))
                continue

            X_train_raw = df.loc[train_mask, feature_cols].copy()
            X_test_raw = df.loc[test_mask, feature_cols].copy()
            y_train = df.loc[train_mask, "home_win"].values
            y_test = df.loc[test_mask, "home_win"].values
            train_years = df.loc[train_mask, "year"].values
            sw = compute_sample_weights(pd.Series(train_years))

            X_train_filled, X_test_filled = fill_missing(X_train_raw, X_test_raw)
            cols = get_shap_top_features(X_train_filled, y_train, sw, feature_cols, top_n)

            X_tr = X_train_filled[cols]
            X_te = X_test_filled[cols]

            odds_test = df.loc[test_mask, "odds_home_prob"].values
            odds_test = np.where(np.isnan(odds_test), 0.55, odds_test)
            pruned_odds.append(odds_test)
            pruned_y.append(y_test)

            for name in models_to_test:
                m = train_model(name, params, X_tr, y_train, sw)
                preds = np.clip(m.predict_proba(X_te)[:, 1], 1e-7, 1 - 1e-7)
                pruned_oof[name].append(preds)

        active = [i for i in range(n_folds) if len(pruned_y[i]) > 0]

        # Equal weight blend: each model + odds
        for combo_name, combo in [("3GBM", ["XGBoost", "LightGBM", "CatBoost"]),
                                   ("4Models", models_to_test)]:
            fold_metrics = []
            for i in active:
                w = 1.0 / (len(combo) + 1)
                blended = np.zeros_like(pruned_y[i], dtype=float)
                for name in combo:
                    blended += w * pruned_oof[name][i]
                blended += w * pruned_odds[i]
                blended = np.clip(blended, 1e-7, 1 - 1e-7)
                fold_metrics.append(compute_metrics(pruned_y[i], blended))
            agg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            diff = agg["log_loss"] - odds_ll
            beat = " BEATS" if diff < -0.0001 else ""
            print(f"    EqualAvg-{combo_name}-{fs_label:5s}: "
                  f"LL={agg['log_loss']:.4f}  Acc={agg['accuracy']:.4f}  "
                  f"vs_odds={diff:+.4f}{beat}")

    return results


# ==========================================================================
#  EXPERIMENT 3: PER-FOLD ANALYSIS
# ==========================================================================

def run_per_fold_analysis(features, feature_cols):
    """Show which years models beat odds and which they don't."""
    print("\n\n" + "=" * 80)
    print("  EXPERIMENT 3: PER-FOLD DIAGNOSTIC")
    print("  (Which years do models add value over odds?)")
    print("=" * 80)

    df = features.copy()
    models_to_test = ["XGBoost", "CatBoost", "RandomForest"]
    params = {
        "xgb": BEST_XGB_PARAMS,
        "cat": BEST_CAT_PARAMS,
        "rf": BEST_RF_PARAMS,
    }

    print(f"\n  {'Year':>6s}  {'Odds LL':>8s}", end="")
    for name in models_to_test:
        print(f"  {name+' LL':>12s}", end="")
    print(f"  {'EqAvg3+Odds':>12s}  {'Best':>12s}")
    print("  " + "-" * 80)

    for fold in FOLDS:
        train_end = fold["train_end"]
        test_year = fold["test_year"]
        train_mask = df["year"] <= train_end
        test_mask = df["year"] == test_year

        if test_mask.sum() == 0:
            continue

        X_train_raw = df.loc[train_mask, feature_cols].copy()
        X_test_raw = df.loc[test_mask, feature_cols].copy()
        y_train = df.loc[train_mask, "home_win"].values
        y_test = df.loc[test_mask, "home_win"].values
        train_years = df.loc[train_mask, "year"].values
        sw = compute_sample_weights(pd.Series(train_years))

        X_train, X_test = fill_missing(X_train_raw, X_test_raw)

        odds_test = df.loc[test_mask, "odds_home_prob"].values
        odds_test = np.where(np.isnan(odds_test), 0.55, odds_test)
        odds_ll = safe_log_loss(y_test, odds_test)

        model_preds = {}
        for name in models_to_test:
            m = train_model(name, params, X_train, y_train, sw)
            preds = np.clip(m.predict_proba(X_test)[:, 1], 1e-7, 1 - 1e-7)
            model_preds[name] = preds

        # Equal blend: 3 models + odds
        w = 0.25
        blended = np.zeros_like(y_test, dtype=float)
        for name in models_to_test:
            blended += w * model_preds[name]
        blended += w * odds_test
        blended = np.clip(blended, 1e-7, 1 - 1e-7)
        blend_ll = safe_log_loss(y_test, blended)

        # Find best
        all_lls = {"Odds": odds_ll, "Blend": blend_ll}
        for name in models_to_test:
            all_lls[name] = safe_log_loss(y_test, model_preds[name])
        best_name = min(all_lls, key=all_lls.get)

        print(f"  {test_year:>6d}  {odds_ll:8.4f}", end="")
        for name in models_to_test:
            ll = all_lls[name]
            marker = "*" if ll < odds_ll else " "
            print(f"  {ll:11.4f}{marker}", end="")
        blend_marker = "*" if blend_ll < odds_ll else " "
        print(f"  {blend_ll:11.4f}{blend_marker}  {best_name:>12s}")


# ==========================================================================
#  MAIN
# ==========================================================================

def main():
    start = time.time()
    print("*" * 80)
    print("*  HONEST EVALUATION EXPERIMENT")
    print("*  Can our models actually beat odds in walk-forward evaluation?")
    print("*  Odds baseline: LL=0.5999, Acc=68.00%")
    print("*" * 80)

    # Load features
    feat_path = FEATURES_DIR / "features_v5_final.parquet"
    if not feat_path.exists():
        print(f"  ERROR: {feat_path} not found. Run run_v5_final_pipeline.py first.")
        sys.exit(1)

    features = pd.read_parquet(feat_path)
    print(f"\n  Loaded: {feat_path.name} ({features.shape[0]} rows x {features.shape[1]} cols)")

    meta_cols = ["year", "round", "home_team", "away_team", "home_win",
                 "home_score", "away_score", "match_id", "venue",
                 "odds_home_prob"]
    feature_cols = [c for c in features.columns if c not in meta_cols
                    and c != "home_win" and features[c].dtype in ["float64", "float32", "int64", "int32"]]
    feature_cols = [c for c in feature_cols if c in features.columns]
    print(f"  Feature columns: {len(feature_cols)}")

    # Run experiments
    run_constrained_blending(features, feature_cols)
    run_feature_pruning(features, feature_cols)
    run_per_fold_analysis(features, feature_cols)

    elapsed = time.time() - start
    print(f"\n\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 80)
    print("  EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
