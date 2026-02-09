"""
Simulated betting P&L for NRL match-winner predictions.

Compares a model's predicted probabilities against bookmaker-implied
probabilities and simulates different staking strategies (flat-stake value
betting, fractional Kelly criterion).

All monetary values are in abstract *units* (default bankroll 1000, default
unit stake 10).  Decimal odds are used throughout (e.g., 2.10 means a $1
bet returns $2.10 on a win).

Key classes and functions
-------------------------
* :class:`BettingSimulator` -- core simulation engine.
* :func:`simulate_season` -- convenience wrapper for a single-season run.
* :func:`plot_cumulative_pnl` -- line chart of cumulative profit/loss.
* :func:`plot_roi_by_season` -- bar chart comparing ROI across seasons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import OUTPUTS_DIR

logger = logging.getLogger(__name__)

FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data container for individual bets
# ---------------------------------------------------------------------------


@dataclass
class BetRecord:
    """Immutable record of a single placed bet."""

    match_index: int
    stake: float
    odds_decimal: float
    model_prob: float
    implied_prob: float
    won: bool
    payout: float
    profit: float
    bankroll_after: float


# ---------------------------------------------------------------------------
# Betting simulator
# ---------------------------------------------------------------------------


class BettingSimulator:
    """Simulate staking strategies against bookmaker odds.

    Parameters
    ----------
    initial_bankroll : float
        Starting bankroll.  Default ``1000``.
    unit_stake : float
        Flat stake per bet when using :meth:`flat_stake_value_bet`.
        Default ``10``.
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        unit_stake: float = 10.0,
    ) -> None:
        if initial_bankroll <= 0:
            raise ValueError("initial_bankroll must be positive.")
        if unit_stake <= 0:
            raise ValueError("unit_stake must be positive.")

        self.initial_bankroll = float(initial_bankroll)
        self.unit_stake = float(unit_stake)

        # Mutable state – reset via reset().
        self._bankroll: float = self.initial_bankroll
        self._bets: list[BetRecord] = []
        self._peak_bankroll: float = self.initial_bankroll

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the simulator to its initial state."""
        self._bankroll = self.initial_bankroll
        self._bets = []
        self._peak_bankroll = self.initial_bankroll

    @property
    def bankroll(self) -> float:
        """Current bankroll."""
        return self._bankroll

    @property
    def bets(self) -> list[BetRecord]:
        """List of all placed bets (chronological order)."""
        return list(self._bets)

    # ------------------------------------------------------------------
    # Staking strategies
    # ------------------------------------------------------------------

    def flat_stake_value_bet(
        self,
        y_prob_model: np.ndarray | pd.Series,
        odds_implied_prob: np.ndarray | pd.Series,
        odds_decimal: np.ndarray | pd.Series,
        y_true: np.ndarray | pd.Series,
        threshold: float = 0.05,
    ) -> list[BetRecord]:
        """Simulate flat-stake value betting.

        A bet is placed whenever the model's probability exceeds the
        bookmaker-implied probability by at least *threshold*.

        Parameters
        ----------
        y_prob_model : array-like
            Model's predicted probability of the outcome being backed.
        odds_implied_prob : array-like
            Bookmaker-implied probability (``1 / odds_decimal``, possibly
            after removing overround).
        odds_decimal : array-like
            Decimal odds for the backed outcome.
        y_true : array-like of {0, 1}
            Ground-truth result (1 = backed outcome won).
        threshold : float
            Minimum edge (``model_prob - implied_prob``) to trigger a bet.
            Default ``0.05`` (5 percentage points).

        Returns
        -------
        list[BetRecord]
            Records of bets placed during this simulation.
        """
        y_prob_model = np.asarray(y_prob_model, dtype=float)
        odds_implied_prob = np.asarray(odds_implied_prob, dtype=float)
        odds_decimal = np.asarray(odds_decimal, dtype=float)
        y_true = np.asarray(y_true, dtype=int)

        self._validate_simulation_inputs(
            y_prob_model, odds_implied_prob, odds_decimal, y_true
        )

        placed: list[BetRecord] = []

        for i in range(len(y_prob_model)):
            edge = y_prob_model[i] - odds_implied_prob[i]
            if edge < threshold:
                continue
            if self._bankroll < self.unit_stake:
                logger.warning(
                    "Bankroll exhausted at bet index %d. Stopping.", i
                )
                break

            stake = self.unit_stake
            record = self._place_bet(
                match_index=i,
                stake=stake,
                odds_decimal=float(odds_decimal[i]),
                model_prob=float(y_prob_model[i]),
                implied_prob=float(odds_implied_prob[i]),
                won=bool(y_true[i] == 1),
            )
            placed.append(record)

        return placed

    def kelly_criterion_bet(
        self,
        y_prob_model: np.ndarray | pd.Series,
        odds_decimal: np.ndarray | pd.Series,
        y_true: np.ndarray | pd.Series,
        fraction: float = 0.25,
        *,
        min_edge: float = 0.0,
    ) -> list[BetRecord]:
        """Simulate fractional Kelly-criterion staking.

        The Kelly stake fraction is::

            f* = (p * (d - 1) - (1 - p)) / (d - 1)

        where *p* is the model probability and *d* is the decimal odds.  The
        actual stake is ``fraction * f* * bankroll`` (fractional Kelly for
        risk reduction).

        Parameters
        ----------
        y_prob_model : array-like
            Model's predicted probability of the backed outcome.
        odds_decimal : array-like
            Decimal odds.
        y_true : array-like of {0, 1}
            Ground-truth outcome.
        fraction : float
            Kelly fraction (e.g. 0.25 = quarter-Kelly).  Default ``0.25``.
        min_edge : float
            Minimum edge (``kelly_fraction``) before placing a bet.
            Default ``0.0``.

        Returns
        -------
        list[BetRecord]
            Records of bets placed during this simulation.
        """
        y_prob_model = np.asarray(y_prob_model, dtype=float)
        odds_decimal = np.asarray(odds_decimal, dtype=float)
        y_true = np.asarray(y_true, dtype=int)

        if len(y_prob_model) != len(odds_decimal) or len(odds_decimal) != len(y_true):
            raise ValueError("All input arrays must have the same length.")
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be in (0, 1].")

        placed: list[BetRecord] = []

        for i in range(len(y_prob_model)):
            p = y_prob_model[i]
            d = odds_decimal[i]

            if d <= 1.0:
                # Degenerate odds – skip.
                continue

            kelly_f = (p * (d - 1.0) - (1.0 - p)) / (d - 1.0)

            if kelly_f <= min_edge:
                continue

            stake = max(fraction * kelly_f * self._bankroll, 0.0)

            # Floor tiny stakes and skip.
            if stake < 0.01:
                continue

            # Never bet more than current bankroll.
            stake = min(stake, self._bankroll)

            implied_prob = 1.0 / d if d > 0 else 0.0

            record = self._place_bet(
                match_index=i,
                stake=stake,
                odds_decimal=float(d),
                model_prob=float(p),
                implied_prob=implied_prob,
                won=bool(y_true[i] == 1),
            )
            placed.append(record)

        return placed

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return a dictionary summarising betting performance.

        Returns
        -------
        dict[str, Any]
            Keys: ``total_bets``, ``wins``, ``losses``, ``win_rate``,
            ``total_staked``, ``total_payout``, ``profit_loss``, ``roi``,
            ``max_drawdown``, ``max_drawdown_pct``, ``sharpe_ratio``,
            ``final_bankroll``.
        """
        if not self._bets:
            return {
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_staked": 0.0,
                "total_payout": 0.0,
                "profit_loss": 0.0,
                "roi": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "final_bankroll": self.initial_bankroll,
            }

        wins = sum(1 for b in self._bets if b.won)
        losses = len(self._bets) - wins
        total_staked = sum(b.stake for b in self._bets)
        total_payout = sum(b.payout for b in self._bets)
        profit_loss = total_payout - total_staked

        roi = (profit_loss / total_staked) * 100.0 if total_staked > 0 else 0.0

        # Max drawdown.
        bankroll_curve = [self.initial_bankroll] + [
            b.bankroll_after for b in self._bets
        ]
        max_drawdown, max_drawdown_pct = self._compute_max_drawdown(bankroll_curve)

        # Sharpe ratio (annualised-ish – per-bet returns).
        per_bet_returns = np.array([b.profit / b.stake for b in self._bets])
        sharpe = self._compute_sharpe(per_bet_returns)

        return {
            "total_bets": len(self._bets),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(self._bets) if self._bets else 0.0,
            "total_staked": round(total_staked, 2),
            "total_payout": round(total_payout, 2),
            "profit_loss": round(profit_loss, 2),
            "roi": round(roi, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe, 4),
            "final_bankroll": round(self._bankroll, 2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _place_bet(
        self,
        match_index: int,
        stake: float,
        odds_decimal: float,
        model_prob: float,
        implied_prob: float,
        won: bool,
    ) -> BetRecord:
        """Execute a single bet and update internal state."""
        self._bankroll -= stake

        if won:
            payout = stake * odds_decimal
            profit = payout - stake
        else:
            payout = 0.0
            profit = -stake

        self._bankroll += payout
        self._peak_bankroll = max(self._peak_bankroll, self._bankroll)

        record = BetRecord(
            match_index=match_index,
            stake=round(stake, 2),
            odds_decimal=odds_decimal,
            model_prob=model_prob,
            implied_prob=implied_prob,
            won=won,
            payout=round(payout, 2),
            profit=round(profit, 2),
            bankroll_after=round(self._bankroll, 2),
        )
        self._bets.append(record)
        return record

    @staticmethod
    def _validate_simulation_inputs(
        y_prob: np.ndarray,
        implied_prob: np.ndarray,
        odds: np.ndarray,
        y_true: np.ndarray,
    ) -> None:
        """Check that all simulation input arrays are conformable."""
        lengths = {len(y_prob), len(implied_prob), len(odds), len(y_true)}
        if len(lengths) != 1:
            raise ValueError(
                "All input arrays must have the same length. "
                f"Got lengths: y_prob={len(y_prob)}, implied_prob={len(implied_prob)}, "
                f"odds={len(odds)}, y_true={len(y_true)}."
            )

    @staticmethod
    def _compute_max_drawdown(
        bankroll_curve: list[float] | np.ndarray,
    ) -> tuple[float, float]:
        """Return (absolute max drawdown, percentage max drawdown)."""
        curve = np.asarray(bankroll_curve, dtype=float)
        if len(curve) < 2:
            return 0.0, 0.0

        peak = np.maximum.accumulate(curve)
        drawdowns = peak - curve
        max_dd = float(drawdowns.max())

        # Percentage relative to the peak at the point of max drawdown.
        idx = int(drawdowns.argmax())
        peak_at_dd = peak[idx]
        max_dd_pct = (max_dd / peak_at_dd * 100.0) if peak_at_dd > 0 else 0.0

        return max_dd, max_dd_pct

    @staticmethod
    def _compute_sharpe(
        per_bet_returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Compute a Sharpe-like ratio from per-bet returns."""
        if len(per_bet_returns) < 2:
            return 0.0
        excess = per_bet_returns - risk_free_rate
        std = float(np.std(excess, ddof=1))
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def simulate_season(
    predictions_df: pd.DataFrame,
    strategy: Literal["flat_value", "kelly"] = "flat_value",
    threshold: float = 0.05,
    kelly_fraction: float = 0.25,
    initial_bankroll: float = 1000.0,
    unit_stake: float = 10.0,
) -> dict[str, Any]:
    """Run a betting simulation on a DataFrame of predictions.

    The DataFrame must contain the following columns:

    * ``y_true`` -- ground-truth result (1 = backed outcome won).
    * ``y_prob`` -- model's predicted probability of the backed outcome.
    * ``odds_decimal`` -- decimal odds for the backed outcome.

    Optionally:

    * ``odds_implied_prob`` -- bookmaker-implied probability.  If absent,
      computed as ``1 / odds_decimal``.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Prediction data for the season.
    strategy : ``"flat_value"`` | ``"kelly"``
        Staking strategy.  Default ``"flat_value"``.
    threshold : float
        Edge threshold for flat-value betting.  Default ``0.05``.
    kelly_fraction : float
        Fractional Kelly multiplier.  Default ``0.25``.
    initial_bankroll : float
        Starting bankroll.  Default ``1000``.
    unit_stake : float
        Flat-stake size.  Default ``10``.

    Returns
    -------
    dict[str, Any]
        Summary dict from :meth:`BettingSimulator.get_summary` plus a
        ``"bets"`` key containing the list of :class:`BetRecord` objects.
    """
    required_cols = {"y_true", "y_prob", "odds_decimal"}
    missing = required_cols - set(predictions_df.columns)
    if missing:
        raise ValueError(
            f"predictions_df is missing required column(s): {sorted(missing)}"
        )

    df = predictions_df.copy()

    if "odds_implied_prob" not in df.columns:
        df["odds_implied_prob"] = 1.0 / df["odds_decimal"]

    sim = BettingSimulator(
        initial_bankroll=initial_bankroll,
        unit_stake=unit_stake,
    )

    if strategy == "flat_value":
        sim.flat_stake_value_bet(
            y_prob_model=df["y_prob"].values,
            odds_implied_prob=df["odds_implied_prob"].values,
            odds_decimal=df["odds_decimal"].values,
            y_true=df["y_true"].values,
            threshold=threshold,
        )
    elif strategy == "kelly":
        sim.kelly_criterion_bet(
            y_prob_model=df["y_prob"].values,
            odds_decimal=df["odds_decimal"].values,
            y_true=df["y_true"].values,
            fraction=kelly_fraction,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'flat_value' or 'kelly'.")

    summary = sim.get_summary()
    summary["bets"] = sim.bets
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_cumulative_pnl(
    results: dict[str, Any],
    title: str = "Cumulative P&L",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot cumulative profit/loss over placed bets.

    Parameters
    ----------
    results : dict
        Output of :func:`simulate_season` (must contain ``"bets"`` key).
    title : str
        Chart title.  Default ``"Cumulative P&L"``.
    save_path : str | Path | None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    bets: list[BetRecord] = results.get("bets", [])
    if not bets:
        logger.warning("No bets to plot.")
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.text(0.5, 0.5, "No bets placed", transform=ax.transAxes, ha="center")
        return fig

    initial = bets[0].bankroll_after - bets[0].profit  # reconstruct initial bankroll
    cumulative = np.cumsum([b.profit for b in bets])
    bet_numbers = np.arange(1, len(bets) + 1)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(bet_numbers, cumulative, linewidth=1.5, color="#1f77b4")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.fill_between(
        bet_numbers,
        cumulative,
        0,
        where=cumulative >= 0,
        alpha=0.15,
        color="green",
        interpolate=True,
    )
    ax.fill_between(
        bet_numbers,
        cumulative,
        0,
        where=cumulative < 0,
        alpha=0.15,
        color="red",
        interpolate=True,
    )

    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Cumulative Profit / Loss ($)")
    ax.set_title(title)

    # Annotate final P&L.
    final_pnl = cumulative[-1]
    colour = "green" if final_pnl >= 0 else "red"
    ax.annotate(
        f"Final: ${final_pnl:+.2f}",
        xy=(bet_numbers[-1], final_pnl),
        fontsize=10,
        fontweight="bold",
        color=colour,
        xytext=(10, 10),
        textcoords="offset points",
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved cumulative P&L figure to %s", save_path)

    return fig


def plot_roi_by_season(
    results_by_year: dict[int, dict[str, Any]],
    title: str = "ROI by Season",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of ROI for each season.

    Parameters
    ----------
    results_by_year : dict[int, dict]
        Mapping from year to the summary dict returned by
        :func:`simulate_season` or :meth:`BettingSimulator.get_summary`.
    title : str
        Chart title.
    save_path : str | Path | None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if not results_by_year:
        logger.warning("No season results to plot.")
        fig, ax = plt.subplots()
        ax.set_title(title)
        return fig

    years = sorted(results_by_year.keys())
    rois = [results_by_year[y].get("roi", 0.0) for y in years]
    colours = ["#2ca02c" if r >= 0 else "#d62728" for r in rois]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar([str(y) for y in years], rois, color=colours, edgecolor="white")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Season")
    ax.set_ylabel("ROI (%)")
    ax.set_title(title)

    # Label each bar.
    for bar, roi in zip(bars, rois):
        y_pos = bar.get_height()
        va = "bottom" if y_pos >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{roi:+.1f}%",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved ROI-by-season figure to %s", save_path)

    return fig
