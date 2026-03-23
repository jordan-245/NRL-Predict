"""
Travel distance features for NRL match prediction.

Uses GPS coordinates from ``config.venues`` to compute how far each team
must travel from their home city to the match venue.

Features produced
-----------------
home_travel_km      : float — distance (km) from home team's city to venue
away_travel_km      : float — distance (km) from away team's city to venue
travel_diff_km      : float — away_travel_km − home_travel_km  (positive =
                      away team travels further)
away_is_interstate  : float (0/1) — away team crossing a state boundary
                      (proxy: away_travel_km > 300 km and not overseas)
away_is_overseas    : float (0/1) — away team travelling internationally;
                      covers NZ Warriors playing in Australia AND Australian
                      teams playing in NZ or other international venues (UK,
                      PNG, etc.)

Overseas detection uses venue longitude rather than raw distance, because
some long domestic routes (e.g. Melbourne → Townsville ≈ 2 070 km) would
exceed a pure km threshold designed to separate Auckland (≈ 2 155 km from
Sydney) from Australian cities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.venues import (
    CITY_COORDS,
    TEAM_HOME_CITY,
    lookup_venue_coords,
    travel_distance_km,
)

# Distance threshold to classify as interstate (km).
_INTERSTATE_KM = 300

# Longitude ranges — Australia roughly 113–154°E, NZ roughly 165–178°E.
# Venues outside 100–160°E are treated as international (NZ, UK, PNG …).
_AUS_LON_MIN = 100.0
_AUS_LON_MAX = 160.0

# Latitude range for New Zealand (used together with longitude).
_NZ_LAT_MIN = -48.0
_NZ_LAT_MAX = -33.0
_NZ_LON_MIN = 165.0
_NZ_LON_MAX = 178.5


def _is_nz_venue(lat: float, lon: float) -> bool:
    return (_NZ_LAT_MIN <= lat <= _NZ_LAT_MAX) and (_NZ_LON_MIN <= lon <= _NZ_LON_MAX)


def _is_aus_venue(lat: float, lon: float) -> bool:
    return _AUS_LON_MIN <= lon <= _AUS_LON_MAX


def _team_is_nz(team: str) -> bool:
    """Return True for the New Zealand Warriors (any alias)."""
    return "warrior" in team.lower() or "new zealand" in team.lower()


def compute_travel_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute home/away travel distance features.

    Parameters
    ----------
    matches : pd.DataFrame
        Match rows. Required columns: ``home_team``, ``away_team``,
        ``venue``.  Other columns are passed through unchanged.

    Returns
    -------
    pd.DataFrame
        Copy of *matches* with five new columns appended:
        ``home_travel_km``, ``away_travel_km``, ``travel_diff_km``,
        ``away_is_interstate``, ``away_is_overseas``.
    """
    print("\n" + "=" * 80)
    print("  FEATURES: COMPUTING TRAVEL DISTANCE FEATURES")
    print("=" * 80)

    df = matches.copy()

    home_travel_list: list[float] = []
    away_travel_list: list[float] = []
    travel_diff_list: list[float] = []
    away_interstate_list: list[float] = []
    away_overseas_list: list[float] = []

    n_resolved = 0  # count rows where venue GPS was resolved

    for _, row in df.iterrows():
        home_team = str(row.get("home_team", "") or "").strip()
        away_team = str(row.get("away_team", "") or "").strip()
        venue = str(row.get("venue", "") or "").strip()

        h_dist = travel_distance_km(home_team, venue)
        a_dist = travel_distance_km(away_team, venue)

        home_travel_list.append(h_dist)
        away_travel_list.append(a_dist)
        travel_diff_list.append(a_dist - h_dist)

        # ---- overseas / interstate detection using venue coords -----------
        venue_coords = lookup_venue_coords(venue)

        if venue_coords is not None:
            n_resolved += 1
            v_lat, v_lon = venue_coords

            # Overseas = venue is not in Australia.
            venue_is_overseas = not _is_aus_venue(v_lat, v_lon)

            if not venue_is_overseas:
                # Australian venue: overseas if away team is NZ Warriors
                away_overseas = float(_team_is_nz(away_team))
            else:
                # Non-Australian venue (NZ, UK, PNG, etc.)
                # All teams are "overseas" at such venues.
                away_overseas = 1.0
        else:
            # Unknown venue — fall back to distance heuristic:
            # Auckland to closest Aus city ~2 155 km; use 2 100 as cutoff.
            away_overseas = float(a_dist >= 2_100)
            venue_is_overseas = away_overseas == 1.0

        # Interstate = significant domestic travel (not overseas)
        away_interstate = float(
            a_dist >= _INTERSTATE_KM and not bool(away_overseas)
        )

        away_overseas_list.append(away_overseas)
        away_interstate_list.append(away_interstate)

    df["home_travel_km"] = home_travel_list
    df["away_travel_km"] = away_travel_list
    df["travel_diff_km"] = travel_diff_list
    df["away_is_interstate"] = away_interstate_list
    df["away_is_overseas"] = away_overseas_list

    n = len(df)
    print(
        f"  home_travel_km  : mean={np.nanmean(home_travel_list):.0f} km  "
        f"(venue resolved in {n_resolved}/{n} rows)"
    )
    print(
        f"  away_travel_km  : mean={np.nanmean(away_travel_list):.0f} km"
    )
    print(
        f"  away_is_interstate : {int(sum(away_interstate_list))} matches "
        f"({100 * sum(away_interstate_list) / max(n, 1):.1f}%)"
    )
    print(
        f"  away_is_overseas   : {int(sum(away_overseas_list))} matches "
        f"({100 * sum(away_overseas_list) / max(n, 1):.1f}%)"
    )
    print(f"  Total new columns: 5")

    return df
