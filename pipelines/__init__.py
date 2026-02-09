"""
Pipeline orchestration scripts for the NRL Match Winner Prediction project.

Scripts
-------
scrape_all
    End-to-end scrape orchestrator: downloads HTML from Rugby League Project,
    parses matches/lineups/ladders/players, loads odds, and saves processed
    parquet files.
build_features
    Raw/processed data -> feature matrix pipeline.  Cleans, links, computes
    Elo, rolling stats, and builds versioned feature sets.
train_and_evaluate
    Full training + walk-forward backtest pipeline.  Runs baselines, classical
    models with optional Optuna search, betting simulation, and generates an
    HTML comparison report.
predict_upcoming
    Generate win-probability predictions for upcoming rounds using a saved
    model from the registry.
"""
