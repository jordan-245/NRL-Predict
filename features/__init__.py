"""
NRL-Predict feature engineering modules.

Each sub-module exposes a ``compute_X_features(matches, ...) -> pd.DataFrame``
function that follows the standard pipeline contract:

* Accepts a ``pd.DataFrame`` of match rows (sorted chronologically).
* Returns a **copy** with new feature columns appended (never mutates input).
* Handles missing data gracefully — unknown values produce NaN or a
  sensible default (0 / False) rather than raising an exception.
* Has **no look-ahead bias** — only information available *before* each
  match is used.

Available modules
-----------------
features.travel
    ``compute_travel_features`` — home/away travel distances and
    interstate / overseas flags derived from ``config.venues``.

features.weather
    ``compute_weather_proxy_features`` — static weather proxies (wet
    season, cold game, hot game) derived from match date and venue
    latitude.

features.early_season
    ``compute_early_season_features`` — early-season data reliability
    dampening (owned by Builder 2).

features.roster_turnover
    ``compute_roster_turnover_features`` — off-season roster continuity
    metrics (owned by Builder 2).

features.opponent_adjusted
    ``compute_opponent_adjusted_features`` — Elo-weighted rolling stats
    (owned by Builder 2).

features.game_context
    ``compute_game_context_features`` — finals pressure and ladder gap
    signals (owned by Builder 2).
"""

from __future__ import annotations

# Re-export the travel module owned by this builder for convenient import.
from features.travel import compute_travel_features  # noqa: F401

__all__ = [
    "compute_travel_features",
]
