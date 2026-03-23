"""
V4.2 Hyperparameter Tuning + Feature Count Optimization
========================================================
Optuna search on the 204-feature set with walk-forward validation.
Also tests optimal feature pruning levels (top-30, 40, 50, 60, 75, 100).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, accuracy_score
from pathlib import Path

import pipelines.v4 as v4

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_PATH = Path("data/features/features_v4.parquet")
FOLDS = v4.FOLDS


def load_features():
    """Load the V4.2 feature matrix."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Run pipelines/v4.py first to generate {FEATURES_PATH}")
    features = pd.read_parquet(FEATURES_PATH)
    _, feature_cols = v4.build_v4_feature_matrix(features)
    return features, feature_cols


def walk_forward_score(features, feature_cols, model_fn, top_n=None):
    """Score a model config with walk-forward validation.
    
    Returns (accuracy, log_loss) averaged across folds.
    """
    accs, lls = [], []
    
    for fold_idx, fold in enumerate(FOLDS):
        train_end = int(fold["train_end"])
        test_year = int(fold["test_year"])
        if test_year > 2025:
            continue
            
        train = features[(features["year"].astype(int) >= 2013) & (features["year"].astype(int) <= train_end)].copy()
        test = features[features["year"].astype(int) == test_year].copy()
        
        if len(test) == 0 or len(train) < 100:
            continue
        
        # Select features
        active_cols = [c for c in feature_cols if c in train.columns]
        
        X_train = train[active_cols].copy()
        y_train = train["home_win"].values
        X_test = test[active_cols].copy()
        y_test = test["home_win"].values
        
        # Feature selection if requested
        if top_n and top_n < len(active_cols):
            # Quick importance-based selection
            selector = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.02, 
                                          random_state=42, n_jobs=-1, verbosity=0)
            selector.fit(X_train, y_train)
            imp = pd.Series(selector.feature_importances_, index=active_cols).sort_values(ascending=False)
            selected = list(imp.head(top_n).index)
            X_train = X_train[selected]
            X_test = X_test[selected]
        
        # Fill missing
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        
        # Train and predict
        model = model_fn()
        model.fit(X_train, y_train)
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict(X_test)
        
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        acc = accuracy_score(y_test, (probs >= 0.5).astype(int))
        ll = log_loss(y_test, probs)
        accs.append(acc)
        lls.append(ll)
    
    return np.mean(accs), np.mean(lls)


def tune_catboost(features, feature_cols, n_trials=80):
    """Tune CatBoost hyperparameters."""
    print("\n  Tuning CatBoost...")
    
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 0.8),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "random_seed": 42,
            "verbose": False,
        }
        
        def model_fn():
            return CatBoostClassifier(**params)
        
        _, ll = walk_forward_score(features, feature_cols, model_fn)
        return ll
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best = study.best_params
    best["random_seed"] = 42
    best["verbose"] = False
    print(f"  Best CatBoost LL: {study.best_value:.4f}")
    print(f"  Params: {best}")
    return best


def tune_xgboost(features, feature_cols, n_trials=80):
    """Tune XGBoost hyperparameters."""
    print("\n  Tuning XGBoost...")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.15, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "gamma": trial.suggest_float("gamma", 0.1, 5.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        
        def model_fn():
            return xgb.XGBClassifier(**params)
        
        _, ll = walk_forward_score(features, feature_cols, model_fn)
        return ll
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best = study.best_params
    best["random_state"] = 42
    best["n_jobs"] = -1
    best["verbosity"] = 0
    print(f"  Best XGBoost LL: {study.best_value:.4f}")
    print(f"  Params: {best}")
    return best


def tune_lightgbm(features, feature_cols, n_trials=80):
    """Tune LightGBM hyperparameters."""
    print("\n  Tuning LightGBM...")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.15, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        
        def model_fn():
            return lgb.LGBMClassifier(**params)
        
        _, ll = walk_forward_score(features, feature_cols, model_fn)
        return ll
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best = study.best_params
    best["random_state"] = 42
    best["n_jobs"] = -1
    best["verbose"] = -1
    print(f"  Best LightGBM LL: {study.best_value:.4f}")
    print(f"  Params: {best}")
    return best


def test_feature_counts(features, feature_cols):
    """Test different feature pruning levels."""
    print("\n  Testing feature count optimization...")
    
    # Use current best CatBoost params
    cat_params = dict(v4.BEST_CAT_PARAMS)
    cat_params["random_seed"] = 42
    cat_params["verbose"] = False
    
    results = {}
    for top_n in [30, 40, 50, 60, 75, 100, None]:
        label = f"top-{top_n}" if top_n else "all"
        
        def model_fn():
            return CatBoostClassifier(**cat_params)
        
        acc, ll = walk_forward_score(features, feature_cols, model_fn, top_n=top_n)
        results[label] = {"acc": acc, "ll": ll}
        print(f"    CatBoost {label:8s}: Acc={acc:.4f}  LL={ll:.4f}")
    
    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=80, help="Optuna trials per model")
    p.add_argument("--skip-tune", action="store_true", help="Skip tuning, just test feature counts")
    args = p.parse_args()
    
    print("=" * 70)
    print("  V4.2 Hyperparameter Tuning + Feature Optimization")
    print("=" * 70)
    
    features, feature_cols = load_features()
    print(f"  Features: {len(features)} rows, {len(feature_cols)} columns")
    
    # Test feature pruning levels first (fast, informative)
    feat_results = test_feature_counts(features, feature_cols)
    
    if not args.skip_tune:
        # Tune each model
        cat_params = tune_catboost(features, feature_cols, n_trials=args.trials)
        xgb_params = tune_xgboost(features, feature_cols, n_trials=args.trials)
        lgb_params = tune_lightgbm(features, feature_cols, n_trials=args.trials)
        
        # Save tuned params
        import json
        output = {
            "catboost": cat_params,
            "xgboost": xgb_params,
            "lightgbm": lgb_params,
            "feature_counts": feat_results,
        }
        
        out_path = Path("outputs/reports/v42_tuned_params.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved tuned params to {out_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  FEATURE COUNT RESULTS")
    print("=" * 70)
    print(f"  {'Level':10s} {'Accuracy':>10s} {'Log Loss':>10s}")
    print("-" * 35)
    for label, m in feat_results.items():
        print(f"  {label:10s} {m['acc']:10.4f} {m['ll']:10.4f}")


if __name__ == "__main__":
    main()
