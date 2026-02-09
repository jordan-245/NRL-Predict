"""
Evaluation module for NRL Match Winner Prediction.

Provides backtesting, metrics, betting simulation, and report generation.

Public API
----------
.. autosummary::
    backtesting.WalkForwardBacktester
    metrics.compute_accuracy
    metrics.compute_log_loss
    metrics.compute_brier_score
    metrics.compute_auc_roc
    metrics.compute_ece
    metrics.compute_all_metrics
    metrics.compare_models
    metrics.print_classification_report
    betting_simulation.BettingSimulator
    betting_simulation.simulate_season
    betting_simulation.plot_cumulative_pnl
    betting_simulation.plot_roi_by_season
    reports.generate_model_comparison_report
    reports.generate_prediction_report
    reports.save_report
    reports.plot_confusion_matrix
    reports.plot_accuracy_by_round
"""

from evaluation.backtesting import WalkForwardBacktester
from evaluation.betting_simulation import (
    BettingSimulator,
    plot_cumulative_pnl,
    plot_roi_by_season,
    simulate_season,
)
from evaluation.metrics import (
    compare_models,
    compute_accuracy,
    compute_all_metrics,
    compute_auc_roc,
    compute_brier_score,
    compute_ece,
    compute_log_loss,
    print_classification_report,
)
from evaluation.reports import (
    generate_model_comparison_report,
    generate_prediction_report,
    plot_accuracy_by_round,
    plot_confusion_matrix,
    save_report,
)

__all__ = [
    # Backtesting
    "WalkForwardBacktester",
    # Metrics
    "compute_accuracy",
    "compute_log_loss",
    "compute_brier_score",
    "compute_auc_roc",
    "compute_ece",
    "compute_all_metrics",
    "compare_models",
    "print_classification_report",
    # Betting simulation
    "BettingSimulator",
    "simulate_season",
    "plot_cumulative_pnl",
    "plot_roi_by_season",
    # Reports
    "generate_model_comparison_report",
    "generate_prediction_report",
    "save_report",
    "plot_confusion_matrix",
    "plot_accuracy_by_round",
]
