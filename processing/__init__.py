"""
Processing package for the NRL Match Winner Prediction project.

Submodules
----------
data_cleaning
    Standardise raw scraped/downloaded data.
data_linking
    Join datasets into a unified master table.
elo
    Elo rating system with configurable K-factor and MOV adjustments.
rolling_stats
    Rolling-window and EWMA team form aggregations.
feature_engineering
    Main feature computation orchestrator.
target_encoding
    Target variable creation and target-encoded categoricals.
"""
