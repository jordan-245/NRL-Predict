"""
Probability calibration for NRL prediction models.

Well-calibrated probabilities are essential for betting simulation and
practical utility.  This module provides:

* **calibrate_platt** -- Platt scaling (logistic regression on model outputs).
* **calibrate_isotonic** -- Isotonic regression calibration.
* **CalibratedModel** -- wrapper that pairs any model with a calibration map.
* **plot_calibration_curve** -- reliability diagram.
* **compute_ece** -- Expected Calibration Error metric.

Typical usage
-------------
>>> from modelling.calibration import CalibratedModel, compute_ece
>>> cal_model = CalibratedModel(base_model, method="platt")
>>> cal_model.fit(X_train, y_train, X_cal, y_cal)
>>> ece = compute_ece(y_test, cal_model.predict_proba(X_test)[:, 1])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from config.settings import OUTPUTS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

_FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. Platt Scaling
# ============================================================================

def calibrate_platt(
    model: Any,
    X_cal: pd.DataFrame | np.ndarray,
    y_cal: ArrayLike,
) -> LogisticRegression:
    """Fit Platt scaling on a calibration set.

    Trains a logistic regression that maps the base model's predicted
    home-win probability to a calibrated probability.

    Parameters
    ----------
    model : estimator
        A fitted model with ``predict_proba(X) -> (n, 2)``.
    X_cal : array-like
        Calibration features.
    y_cal : array-like
        Binary target for the calibration set.

    Returns
    -------
    LogisticRegression
        Fitted calibrator.  Input: raw ``p_home`` column vector;
        output: calibrated probability.
    """
    y_np = np.asarray(y_cal, dtype=int)
    raw_proba = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1e10,  # Effectively unregularised.
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    calibrator.fit(raw_proba, y_np)

    logger.info(
        "Platt calibrator fitted on %d samples. "
        "Slope=%.4f, intercept=%.4f.",
        len(y_np),
        calibrator.coef_[0, 0],
        calibrator.intercept_[0],
    )
    return calibrator


# ============================================================================
# 2. Isotonic Regression Calibration
# ============================================================================

def calibrate_isotonic(
    model: Any,
    X_cal: pd.DataFrame | np.ndarray,
    y_cal: ArrayLike,
) -> IsotonicRegression:
    """Fit isotonic regression calibration on a calibration set.

    Isotonic regression provides a non-parametric monotonic mapping from
    raw probabilities to calibrated probabilities.

    Parameters
    ----------
    model : estimator
        A fitted model with ``predict_proba(X) -> (n, 2)``.
    X_cal : array-like
        Calibration features.
    y_cal : array-like
        Binary target for the calibration set.

    Returns
    -------
    IsotonicRegression
        Fitted calibrator.  Call ``calibrator.predict(p_raw)`` to obtain
        calibrated probabilities.
    """
    y_np = np.asarray(y_cal, dtype=float)
    raw_proba = model.predict_proba(X_cal)[:, 1]

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    calibrator.fit(raw_proba, y_np)

    logger.info(
        "Isotonic calibrator fitted on %d samples.",
        len(y_np),
    )
    return calibrator


# ============================================================================
# 3. CalibratedModel wrapper
# ============================================================================

class CalibratedModel:
    """Wrap a base model with a calibration layer.

    The wrapper first obtains raw probabilities from the base model, then
    passes them through a calibrator (Platt or isotonic).

    Parameters
    ----------
    base_model : estimator
        Any model with ``fit``, ``predict``, ``predict_proba``.
    method : {"platt", "isotonic"}
        Calibration strategy.

    Examples
    --------
    >>> cal = CalibratedModel(my_xgb_model, method="platt")
    >>> cal.fit(X_train, y_train, X_cal=X_cal, y_cal=y_cal)
    >>> proba = cal.predict_proba(X_test)
    """

    def __init__(
        self,
        base_model: Any,
        method: Literal["platt", "isotonic"] = "platt",
    ) -> None:
        if method not in ("platt", "isotonic"):
            raise ValueError(
                f"method must be 'platt' or 'isotonic', got '{method}'"
            )
        self.base_model = base_model
        self.method = method
        self.calibrator_: LogisticRegression | IsotonicRegression | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: ArrayLike,
        X_cal: pd.DataFrame | np.ndarray | None = None,
        y_cal: ArrayLike | None = None,
        refit_base: bool = True,
    ) -> "CalibratedModel":
        """Fit the base model and calibrator.

        If no explicit calibration set is provided, the last 20 % of the
        training data is held out for calibration.

        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training target.
        X_cal : array-like, optional
            Calibration features.  If ``None``, split from training data.
        y_cal : array-like, optional
            Calibration target.
        refit_base : bool
            Whether to re-fit the base model.  Set ``False`` if the model
            is already trained.

        Returns
        -------
        self
        """
        y_train_np = np.asarray(y_train, dtype=int)

        # Split calibration set if not provided.
        if X_cal is None or y_cal is None:
            n = len(y_train_np)
            split = int(n * 0.80)
            if isinstance(X_train, pd.DataFrame):
                X_fit, X_cal_split = X_train.iloc[:split], X_train.iloc[split:]
            else:
                X_arr = np.asarray(X_train)
                X_fit, X_cal_split = X_arr[:split], X_arr[split:]
            y_fit = y_train_np[:split]
            y_cal_split = y_train_np[split:]
        else:
            X_fit = X_train
            y_fit = y_train_np
            X_cal_split = X_cal
            y_cal_split = np.asarray(y_cal, dtype=int)

        # Fit base model.
        if refit_base:
            logger.info("CalibratedModel: fitting base model.")
            self.base_model.fit(X_fit, y_fit)

        # Fit calibrator.
        if self.method == "platt":
            self.calibrator_ = calibrate_platt(
                self.base_model, X_cal_split, y_cal_split
            )
        else:
            self.calibrator_ = calibrate_isotonic(
                self.base_model, X_cal_split, y_cal_split
            )

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return calibrated class probabilities.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            ``[P(away_win), P(home_win)]`` after calibration.
        """
        if not self.is_fitted_:
            raise RuntimeError("CalibratedModel has not been fitted yet.")

        raw_p = self.base_model.predict_proba(X)[:, 1]

        if self.method == "platt":
            cal_p = self.calibrator_.predict_proba(
                raw_p.reshape(-1, 1)
            )[:, 1]
        else:
            cal_p = self.calibrator_.predict(raw_p)

        # Clip to avoid numerical issues.
        cal_p = np.clip(cal_p, 1e-15, 1.0 - 1e-15)
        return np.column_stack([1.0 - cal_p, cal_p])

    # ------------------------------------------------------------------
    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return predicted class labels from calibrated probabilities."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        return {"method": self.method}

    def set_params(self, **params: Any) -> "CalibratedModel":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        base = type(self.base_model).__name__
        return f"CalibratedModel(base={base}, method='{self.method}')"


# ============================================================================
# 4. Calibration Diagnostics
# ============================================================================

def compute_ece(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the average absolute difference between predicted
    confidence and actual accuracy across probability bins.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels.
    y_prob : array-like
        Predicted probabilities for the positive class (home win).
    n_bins : int
        Number of equally-spaced bins.

    Returns
    -------
    float
        The ECE value (lower is better; 0.0 is perfectly calibrated).
    """
    y_true_np = np.asarray(y_true, dtype=float)
    y_prob_np = np.asarray(y_prob, dtype=float)
    n = len(y_true_np)

    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob_np > lo) & (y_prob_np <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_confidence = y_prob_np[mask].mean()
        avg_accuracy = y_true_np[mask].mean()
        ece += (count / n) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def plot_calibration_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    model_name: str = "Model",
    save_path: str | Path | None = None,
    show: bool = False,
) -> "matplotlib.figure.Figure":
    """Plot a reliability diagram (calibration curve).

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins.
    model_name : str
        Label for the plot legend.
    save_path : str or Path, optional
        File path to save the figure.  If ``None``, saves to
        ``outputs/figures/calibration_curve.png``.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    y_true_np = np.asarray(y_true, dtype=float)
    y_prob_np = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres: list[float] = []
    bin_accuracies: list[float] = []
    bin_counts: list[int] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob_np > lo) & (y_prob_np <= hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count == 0:
            continue
        bin_centres.append((lo + hi) / 2.0)
        bin_accuracies.append(float(y_true_np[mask].mean()))

    ece = compute_ece(y_true_np, y_prob_np, n_bins)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Reliability diagram.
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax1.plot(
        bin_centres,
        bin_accuracies,
        "s-",
        color="#1f77b4",
        label=f"{model_name} (ECE={ece:.4f})",
    )
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Calibration Curve (Reliability Diagram)")
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions.
    ax2.bar(
        (bin_edges[:-1] + bin_edges[1:]) / 2.0,
        bin_counts,
        width=1.0 / n_bins * 0.9,
        color="#1f77b4",
        alpha=0.6,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is None:
        save_path = _FIGURES_DIR / "calibration_curve.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Calibration curve saved to %s", save_path)

    if show:
        plt.show()

    plt.close(fig)
    return fig
