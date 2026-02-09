"""
Evaluation report generation for NRL match-winner prediction models.

Produces HTML reports, calibration plots, confusion matrices, and per-round
accuracy charts.  All figures are saved to ``outputs/figures/`` and HTML
reports to ``outputs/reports/``.

Uses matplotlib + seaborn for static plots.
"""

from __future__ import annotations

import html as html_module
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from config.settings import OUTPUTS_DIR
from evaluation.metrics import compare_models, compute_all_metrics

logger = logging.getLogger(__name__)

REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


def generate_model_comparison_report(
    all_results: dict[str, dict[str, Any]],
    *,
    predictions_by_model: dict[str, pd.DataFrame] | None = None,
    models: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    betting_results_by_model: dict[str, dict[str, Any]] | None = None,
    betting_results_by_year: dict[str, dict[int, dict[str, Any]]] | None = None,
    save: bool = True,
    filename: str = "model_comparison_report.html",
) -> str:
    """Create an HTML report comparing multiple models.

    Parameters
    ----------
    all_results : dict[str, dict[str, Any]]
        Outer key = model name; inner dict contains metric values (the output
        of :func:`evaluation.metrics.compute_all_metrics`) and optionally
        ``"per_year"`` (a DataFrame of per-year metrics).
    predictions_by_model : dict[str, pd.DataFrame] | None
        Model name -> DataFrame with ``y_true``, ``y_pred``, ``y_prob``
        columns.  Used to produce calibration and confusion-matrix plots.
    models : dict[str, Any] | None
        Model name -> fitted model object.  Used to extract feature
        importances from tree-based models.
    feature_names : list[str] | None
        Feature names (required when *models* is provided for importance
        plots).
    betting_results_by_model : dict[str, dict] | None
        Model name -> betting summary dict.
    betting_results_by_year : dict[str, dict[int, dict]] | None
        Model name -> {year: betting summary}.
    save : bool
        Persist the report to disk.  Default ``True``.
    filename : str
        Output file name.

    Returns
    -------
    str
        The generated HTML content.
    """
    sections: list[str] = []
    generated_figures: list[Path] = []

    # --- Header -----------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections.append(_html_header(timestamp))

    # --- Model comparison table -------------------------------------------
    metrics_only = {
        name: {k: v for k, v in vals.items() if isinstance(v, (int, float))}
        for name, vals in all_results.items()
    }

    # Merge in betting ROI if available.
    if betting_results_by_model:
        for name, bsummary in betting_results_by_model.items():
            if name in metrics_only:
                metrics_only[name]["simulated_roi"] = bsummary.get("roi", float("nan"))

    comparison_df = compare_models(metrics_only)
    sections.append("<h2>Model Comparison</h2>")
    sections.append(_df_to_html_table(comparison_df.reset_index()))

    # --- Calibration plots ------------------------------------------------
    if predictions_by_model:
        sections.append("<h2>Calibration Plots</h2>")
        for model_name, preds_df in predictions_by_model.items():
            if "y_true" not in preds_df.columns or "y_prob" not in preds_df.columns:
                continue
            fig_path = FIGURES_DIR / f"calibration_{_safe_filename(model_name)}.png"
            fig = _plot_calibration(
                preds_df["y_true"].values,
                preds_df["y_prob"].values,
                model_name=model_name,
                save_path=fig_path,
            )
            plt.close(fig)
            generated_figures.append(fig_path)
            sections.append(
                f'<h3>{html_module.escape(model_name)}</h3>'
                f'<img src="../figures/{fig_path.name}" width="600">'
            )

    # --- Feature importance charts ----------------------------------------
    if models and feature_names:
        sections.append("<h2>Feature Importance</h2>")
        for model_name, model_obj in models.items():
            importances = _extract_feature_importances(model_obj, feature_names)
            if importances is None:
                continue
            fig_path = FIGURES_DIR / f"importance_{_safe_filename(model_name)}.png"
            fig = _plot_feature_importance(
                importances, model_name=model_name, save_path=fig_path
            )
            plt.close(fig)
            generated_figures.append(fig_path)
            sections.append(
                f'<h3>{html_module.escape(model_name)}</h3>'
                f'<img src="../figures/{fig_path.name}" width="700">'
            )

    # --- Betting P&L chart ------------------------------------------------
    if betting_results_by_model:
        sections.append("<h2>Betting Simulation</h2>")
        for model_name, bsummary in betting_results_by_model.items():
            bets = bsummary.get("bets", [])
            if bets:
                from evaluation.betting_simulation import plot_cumulative_pnl

                fig_path = FIGURES_DIR / f"pnl_{_safe_filename(model_name)}.png"
                fig = plot_cumulative_pnl(
                    bsummary,
                    title=f"Cumulative P&L – {model_name}",
                    save_path=fig_path,
                )
                plt.close(fig)
                generated_figures.append(fig_path)
                sections.append(
                    f'<h3>{html_module.escape(model_name)}</h3>'
                    f'<img src="../figures/{fig_path.name}" width="700">'
                )

    # --- Per-season breakdown ---------------------------------------------
    per_year_data_found = False
    for name, vals in all_results.items():
        if "per_year" in vals and isinstance(vals["per_year"], pd.DataFrame):
            if not per_year_data_found:
                sections.append("<h2>Per-Season Breakdown</h2>")
                per_year_data_found = True
            sections.append(f"<h3>{html_module.escape(name)}</h3>")
            sections.append(_df_to_html_table(vals["per_year"].reset_index()))

    # --- Footer -----------------------------------------------------------
    sections.append(_html_footer())

    html_content = "\n".join(sections)

    if save:
        save_report(html_content, filename, format="html")

    return html_content


def generate_prediction_report(
    predictions_df: pd.DataFrame,
    round_num: int | str,
    year: int,
    *,
    save: bool = True,
) -> str:
    """Generate a formatted prediction output for upcoming matches.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain at minimum: ``home_team``, ``away_team``, ``y_prob``.
        Optional: ``y_pred``, ``odds_decimal``, ``model_edge``.
    round_num : int | str
        Round identifier.
    year : int
        Season year.
    save : bool
        Persist to disk.  Default ``True``.

    Returns
    -------
    str
        The generated HTML content.
    """
    required = {"home_team", "away_team", "y_prob"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions_df missing column(s): {sorted(missing)}")

    sections: list[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections.append(_html_header(timestamp, title=f"Predictions – {year} Round {round_num}"))
    sections.append(f"<h2>Round {round_num}, {year}</h2>")
    sections.append(f"<p>Generated: {timestamp}</p>")

    # Build display table.
    display_df = predictions_df[["home_team", "away_team", "y_prob"]].copy()
    display_df["away_prob"] = 1.0 - display_df["y_prob"]
    display_df.rename(
        columns={
            "home_team": "Home",
            "away_team": "Away",
            "y_prob": "Home Win %",
            "away_prob": "Away Win %",
        },
        inplace=True,
    )

    # Predicted winner column.
    display_df["Predicted Winner"] = np.where(
        display_df["Home Win %"] >= 0.5,
        display_df["Home"],
        display_df["Away"],
    )
    display_df["Confidence"] = display_df[["Home Win %", "Away Win %"]].max(axis=1)
    display_df["Home Win %"] = (display_df["Home Win %"] * 100).round(1).astype(str) + "%"
    display_df["Away Win %"] = (display_df["Away Win %"] * 100).round(1).astype(str) + "%"
    display_df["Confidence"] = (display_df["Confidence"] * 100).round(1).astype(str) + "%"

    sections.append(_df_to_html_table(display_df))
    sections.append(_html_footer())

    html_content = "\n".join(sections)

    if save:
        fname = f"predictions_{year}_round_{round_num}.html"
        save_report(html_content, fname, format="html")

    return html_content


# ---------------------------------------------------------------------------
# Save / persist
# ---------------------------------------------------------------------------


def save_report(
    content: str,
    filename: str,
    format: str = "html",
) -> Path:
    """Write report content to ``outputs/reports/<filename>``.

    Parameters
    ----------
    content : str
        The report body (HTML string, plain text, etc.).
    filename : str
        Output file name.
    format : str
        ``"html"`` or ``"txt"``.  Used only for logging clarity.

    Returns
    -------
    Path
        The absolute path to the saved file.
    """
    out_path = REPORTS_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    logger.info("Saved %s report to %s", format.upper(), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth labels.
    y_pred : array-like of {0, 1}
        Predicted labels.
    model_name : str
        Label for the title.
    save_path : str | Path | None
        Save location.  Falls back to ``outputs/figures/`` if ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(6, 5))

    display_labels = ["Away Win", "Home Win"]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=display_labels
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)

    title = "Confusion Matrix"
    if model_name:
        title += f" – {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")

    fig.tight_layout()

    save_path = _resolve_figure_path(save_path, f"confusion_{_safe_filename(model_name)}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved confusion matrix to %s", save_path)

    return fig


def plot_accuracy_by_round(
    results_df: pd.DataFrame,
    *,
    round_col: str = "round",
    accuracy_col: str = "accuracy",
    model_name: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Line chart of accuracy across rounds within a season.

    Parameters
    ----------
    results_df : pd.DataFrame
        Per-round results (output of
        :meth:`WalkForwardBacktester.run_round_by_round`).
    round_col : str
        Column containing round identifiers.
    accuracy_col : str
        Column containing accuracy values.
    model_name : str
        Label for the title.
    save_path : str | Path | None
        Save location.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if round_col not in results_df.columns or accuracy_col not in results_df.columns:
        raise ValueError(
            f"results_df must contain '{round_col}' and '{accuracy_col}' columns."
        )

    df = results_df.sort_values(round_col)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df[round_col].astype(str),
        df[accuracy_col],
        marker="o",
        markersize=5,
        linewidth=1.5,
        color="#1f77b4",
        label="Accuracy",
    )

    # Rolling average if enough data.
    if len(df) >= 5:
        rolling = df[accuracy_col].rolling(window=5, min_periods=1).mean()
        ax.plot(
            df[round_col].astype(str),
            rolling,
            linewidth=2,
            linestyle="--",
            color="#ff7f0e",
            label="5-round rolling avg",
        )

    # Reference line at 50%.
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, label="50% baseline")

    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    title = "Accuracy by Round"
    if model_name:
        title += f" – {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)

    # Rotate x-labels if there are many rounds.
    if len(df) > 15:
        plt.xticks(rotation=45, ha="right")

    fig.tight_layout()

    save_path = _resolve_figure_path(save_path, f"accuracy_by_round_{_safe_filename(model_name)}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved accuracy-by-round figure to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "",
    n_bins: int = 10,
    save_path: Path | None = None,
) -> plt.Figure:
    """Internal: produce a calibration (reliability) diagram."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: reliability diagram.
    ax_cal = axes[0]
    try:
        fraction_pos, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
    except ValueError:
        fraction_pos, mean_predicted = np.array([]), np.array([])

    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfectly calibrated")
    if len(fraction_pos) > 0:
        ax_cal.plot(mean_predicted, fraction_pos, "s-", label=model_name or "Model")
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Fraction of positives")
    ax_cal.set_title("Calibration Curve")
    ax_cal.legend(loc="lower right")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)

    # Right panel: histogram of predicted probabilities.
    ax_hist = axes[1]
    ax_hist.hist(y_prob, bins=20, edgecolor="white", alpha=0.8, color="#1f77b4")
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Prediction Distribution")
    ax_hist.set_xlim(0, 1)

    fig.suptitle(
        f"Calibration – {model_name}" if model_name else "Calibration",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved calibration plot to %s", save_path)

    return fig


def _plot_feature_importance(
    importances: pd.Series,
    model_name: str = "",
    top_n: int = 20,
    save_path: Path | None = None,
) -> plt.Figure:
    """Internal: horizontal bar chart of top feature importances."""
    importances = importances.sort_values(ascending=True).tail(top_n)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(importances))))

    ax.barh(importances.index.astype(str), importances.values, color="#2ca02c", edgecolor="white")
    ax.set_xlabel("Importance")
    title = "Feature Importance"
    if model_name:
        title += f" – {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature importance plot to %s", save_path)

    return fig


def _extract_feature_importances(
    model: Any,
    feature_names: list[str],
) -> pd.Series | None:
    """Try to extract feature importances from a model object.

    Supports sklearn tree-based models (``feature_importances_``), linear
    models (``coef_``), and XGBoost/LightGBM/CatBoost wrappers.

    Returns ``None`` if importances cannot be determined.
    """
    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            importances = np.abs(coef[0])
        else:
            importances = np.abs(coef)
    else:
        return None

    if importances is not None and len(importances) == len(feature_names):
        return pd.Series(importances, index=feature_names, name="importance")
    return None


def _resolve_figure_path(
    explicit_path: str | Path | None,
    default_filename: str,
) -> Path:
    """Return a concrete save path, falling back to FIGURES_DIR."""
    if explicit_path is not None:
        p = Path(explicit_path)
    else:
        p = FIGURES_DIR / default_filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _safe_filename(name: str) -> str:
    """Convert a model name to a filesystem-safe string."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_").lower()


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------


def _html_header(
    timestamp: str,
    title: str = "NRL Prediction — Model Evaluation Report",
) -> str:
    """Return the opening HTML boilerplate."""
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_module.escape(title)}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    max-width: 1100px;
    margin: 2rem auto;
    padding: 0 1rem;
    color: #222;
    background: #fafafa;
  }}
  h1 {{ border-bottom: 2px solid #1f77b4; padding-bottom: 0.3em; }}
  h2 {{ color: #1f77b4; margin-top: 2rem; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
    font-size: 0.9rem;
  }}
  th, td {{
    border: 1px solid #ddd;
    padding: 0.5rem 0.75rem;
    text-align: right;
  }}
  th {{ background: #1f77b4; color: #fff; text-align: center; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  td:first-child, th:first-child {{ text-align: left; }}
  img {{ max-width: 100%; height: auto; margin: 0.5rem 0; }}
  .timestamp {{ color: #888; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>{html_module.escape(title)}</h1>
<p class="timestamp">Generated: {html_module.escape(timestamp)}</p>
"""


def _html_footer() -> str:
    """Return closing HTML boilerplate."""
    return """\
<hr>
<p class="timestamp">NRL Match Winner Prediction — evaluation module</p>
</body>
</html>
"""


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a styled HTML table string."""
    # Format float columns.
    formatters: dict[str, Any] = {}
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        formatters[col] = lambda x: f"{x:.4f}" if pd.notna(x) else ""

    return df.to_html(
        index=False,
        float_format="%.4f",
        border=0,
        classes="comparison-table",
        na_rep="—",
    )
