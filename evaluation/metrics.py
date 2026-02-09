"""
Evaluation metrics for NRL match-winner prediction models.

Provides individual metric functions (accuracy, log-loss, Brier score, AUC-ROC,
Expected Calibration Error) as well as convenience helpers for computing all
metrics at once, comparing multiple models, and printing a classification report.

All probability arguments (``y_prob``) are expected to be the predicted
probability of the *positive* class (home-team win).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve  # noqa: F401 – re-exported
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return classification accuracy (proportion of correct predictions).

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth binary labels.
    y_pred : array-like of {0, 1}
        Predicted binary labels.

    Returns
    -------
    float
        Accuracy in the range [0, 1].

    Raises
    ------
    ValueError
        If the input arrays are empty or have mismatched lengths.
    """
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    return float(accuracy_score(y_true, y_pred))


def compute_log_loss(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    eps: float = 1e-15,
) -> float:
    """Return the log-loss (cross-entropy) of predicted probabilities.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth binary labels.
    y_prob : array-like of float
        Predicted probability of the positive class.
    eps : float, optional
        Clipping bound to avoid log(0). Default ``1e-15``.

    Returns
    -------
    float
        Log-loss value (lower is better).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    _check_lengths(y_true_arr, y_prob_arr)
    y_prob_arr = np.clip(y_prob_arr, eps, 1.0 - eps)
    return float(log_loss(y_true_arr, y_prob_arr))


def compute_brier_score(
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> float:
    """Return the Brier score (mean squared error of probabilities).

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth binary labels.
    y_prob : array-like of float
        Predicted probability of the positive class.

    Returns
    -------
    float
        Brier score in [0, 1] (lower is better).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    _check_lengths(y_true_arr, y_prob_arr)
    return float(brier_score_loss(y_true_arr, y_prob_arr))


def compute_auc_roc(
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> float:
    """Return the area under the ROC curve.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth binary labels.
    y_prob : array-like of float
        Predicted probability of the positive class.

    Returns
    -------
    float
        AUC-ROC in [0, 1] (higher is better).  Returns ``NaN`` when only a
        single class is present in *y_true*.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    _check_lengths(y_true_arr, y_prob_arr)

    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) < 2:
        logger.warning(
            "AUC-ROC is undefined when only one class is present in y_true. "
            "Returning NaN."
        )
        return float("nan")

    return float(roc_auc_score(y_true_arr, y_prob_arr))


def compute_ece(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    Partitions predictions into *n_bins* equally-spaced bins by predicted
    probability and computes the weighted average of the absolute difference
    between mean predicted probability and observed frequency in each bin.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth binary labels.
    y_prob : array-like of float
        Predicted probability of the positive class.
    n_bins : int, optional
        Number of calibration bins. Default ``10``.

    Returns
    -------
    float
        ECE in [0, 1] (lower is better).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    _check_lengths(y_true_arr, y_prob_arr)

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true_arr)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include right edge for the last bin to capture p == 1.0.
        if hi == bin_edges[-1]:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        else:
            mask = (y_prob_arr >= lo) & (y_prob_arr < hi)

        bin_size = mask.sum()
        if bin_size == 0:
            continue

        bin_accuracy = y_true_arr[mask].mean()
        bin_confidence = y_prob_arr[mask].mean()
        ece += (bin_size / total) * abs(bin_accuracy - bin_confidence)

    return float(ece)


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def compute_all_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute all available metrics and return them as a dictionary.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth binary labels.
    y_pred : array-like of {0, 1}
        Predicted binary labels (from a threshold on *y_prob* or otherwise).
    y_prob : array-like of float
        Predicted probability of the positive class.
    n_bins : int, optional
        Number of bins for ECE. Default ``10``.

    Returns
    -------
    dict[str, float]
        Keys: ``accuracy``, ``log_loss``, ``brier_score``, ``auc_roc``,
        ``ece``.
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "log_loss": compute_log_loss(y_true, y_prob),
        "brier_score": compute_brier_score(y_true, y_prob),
        "auc_roc": compute_auc_roc(y_true, y_prob),
        "ece": compute_ece(y_true, y_prob, n_bins=n_bins),
    }


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a comparison table from per-model metric dictionaries.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Outer key is the model name; inner dict is the output of
        :func:`compute_all_metrics` (or any dict of metric-name -> value).

    Returns
    -------
    pd.DataFrame
        Indexed by model name, one column per metric, sorted by accuracy
        descending.

    Raises
    ------
    ValueError
        If *results* is empty.
    """
    if not results:
        raise ValueError("results dict must contain at least one model.")

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "model"

    # Sort by accuracy descending (primary), then log-loss ascending.
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "accuracy" in df.columns:
        sort_cols.append("accuracy")
        ascending.append(False)
    if "log_loss" in df.columns:
        sort_cols.append("log_loss")
        ascending.append(True)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)

    return df


def print_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    model_name: str = "",
) -> str:
    """Print and return a sklearn-style classification report.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth binary labels.
    y_pred : array-like of {0, 1}
        Predicted binary labels.
    model_name : str, optional
        Label to include in the header.

    Returns
    -------
    str
        The formatted classification report text.
    """
    y_true, y_pred = _validate_arrays(y_true, y_pred)

    header = f"Classification Report{f' – {model_name}' if model_name else ''}"
    separator = "=" * len(header)
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=["Away Win", "Home Win"],
        zero_division=0,
    )
    full_report = f"\n{separator}\n{header}\n{separator}\n{report_text}"
    print(full_report)
    return full_report


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _validate_arrays(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to numpy arrays and validate lengths."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    _check_lengths(y_true_arr, y_pred_arr)
    return y_true_arr, y_pred_arr


def _check_lengths(a: np.ndarray, b: np.ndarray) -> None:
    """Raise if arrays have different lengths or are empty."""
    if len(a) == 0:
        raise ValueError("Input arrays must not be empty.")
    if len(a) != len(b):
        raise ValueError(
            f"Array length mismatch: {len(a)} vs {len(b)}."
        )
