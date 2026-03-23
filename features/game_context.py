"""
Game Context Features
======================
Captures motivational and contextual factors that affect match outcomes:

- Finals race pressure: how desperately a team needs to win to make finals
- Elimination game flag: win-or-go-home finals matches
- Nothing-to-lose differential: large ladder gaps where the underdog plays freely

Features added (4):
  - home_finals_pressure, away_finals_pressure
  - is_elimination
  - nothing_to_lose_diff
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Total regular-season rounds in NRL
REGULAR_SEASON_ROUNDS = 27

# Elimination-style finals rounds (these round labels indicate do-or-die)
_ELIMINATION_ROUNDS = {"qualifying", "eliminat", "semi", "prelim"}


def compute_game_context_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add game-context motivational features.

    Parameters
    ----------
    matches : pd.DataFrame
        Main matches DataFrame with ``round_number``, ``home_ladder_pos``,
        ``away_ladder_pos``, ``is_finals`` columns.

    Returns
    -------
    pd.DataFrame
        matches with 4 new context feature columns.
    """
    print("\n" + "=" * 80)
    print("  V4.1: COMPUTING GAME CONTEXT FEATURES")
    print("=" * 80)

    df = matches.copy()

    # ── Finals pressure ──────────────────────────────────────────────────
    # Positive = team is outside top 8, pressure to win; negative = safely in
    # Scale by remaining rounds: more pressure when fewer rounds left
    rnd = pd.to_numeric(df.get("round_number"), errors="coerce").fillna(1)
    remaining_frac = np.clip(1.0 - rnd / REGULAR_SEASON_ROUNDS, 0.0, 1.0)

    h_pos = pd.to_numeric(df.get("home_ladder_pos"), errors="coerce").fillna(8)
    a_pos = pd.to_numeric(df.get("away_ladder_pos"), errors="coerce").fillna(8)

    # (ladder_pos - 8): positive means outside top 8, negative means inside
    # Multiply by (1 - remaining_frac) so pressure grows as season progresses
    season_progress = 1.0 - remaining_frac  # 0 early, 1 late
    df["home_finals_pressure"] = (h_pos - 8.0) * season_progress
    df["away_finals_pressure"] = (a_pos - 8.0) * season_progress

    # ── Elimination game ─────────────────────────────────────────────────
    # Flag matches in elimination-style finals rounds
    is_finals = pd.to_numeric(df.get("is_finals"), errors="coerce").fillna(0)
    round_str = df["round"].astype(str).str.lower()

    is_elim = np.zeros(len(df), dtype=float)
    for label in _ELIMINATION_ROUNDS:
        is_elim = np.where(round_str.str.contains(label, na=False), 1.0, is_elim)
    # Also flag all finals games (conservative — most finals are high-stakes)
    is_elim = np.where(is_finals == 1, np.maximum(is_elim, 0.5), is_elim)

    df["is_elimination"] = is_elim

    # ── Nothing-to-lose differential ─────────────────────────────────────
    # Large ladder gap means the lower-ranked team has nothing to lose
    # Absolute difference in positions — higher = more asymmetric motivation
    df["nothing_to_lose_diff"] = np.abs(h_pos - a_pos)

    n_with_pressure = (df["home_finals_pressure"].abs() > 0.1).sum()
    n_elim = (df["is_elimination"] > 0).sum()
    print(f"  Added 4 game context features")
    print(f"    Finals pressure active: {n_with_pressure}/{len(df)} games")
    print(f"    Elimination games: {n_elim}/{len(df)} games")

    return df
