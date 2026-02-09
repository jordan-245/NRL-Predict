"""
Modelling module for NRL match winner prediction.

Submodules
----------
baseline_models
    Simple baselines: home-always, ladder, odds-implied, Elo.
classical_models
    Factory functions for LogReg, Random Forest, XGBoost, LightGBM, CatBoost.
neural_models
    PyTorch MLP and LSTM architectures with sklearn-compatible training wrapper.
ensemble
    Voting, stacking, and odds-blending ensembles.
hyperparameter_search
    Optuna-based hyperparameter optimisation.
calibration
    Probability calibration (Platt, isotonic) and ECE metric.
model_registry
    Save / load / version models with metadata.
interpretability
    SHAP values, feature importance, partial dependence plots.
"""
