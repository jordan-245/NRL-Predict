"""
Weather Proxy Features
=======================
Estimates weather conditions from date + venue latitude. No external API needed.

Features produced:
  - is_wet_season  : month in [1,2,3,11,12] AND venue lat < -30 (Sydney/Melb storms)
  - is_cold_game   : month in [6,7,8]       AND venue lat < -33 (cold southern venues)
  - is_hot_game    : month in [12,1,2]      AND venue lat > -25 (tropical QLD/NZ)

Requires venue GPS coordinates from config.venues (VENUE_COORDS dict).
Falls back gracefully if config.venues is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Try to import venue coordinates from Builder 1's module ─────────────────
try:
    from config.venues import VENUE_COORDS  # type: ignore[import]
    _HAVE_COORDS = True
except ImportError:
    # Builder 1's config/venues.py not yet available — use inline fallback
    _HAVE_COORDS = False
    VENUE_COORDS: dict[str, tuple[float, float]] = {}

# Inline fallback coordinates for major venues (lat, lon)
# Used only when config.venues is unavailable
_FALLBACK_COORDS: dict[str, tuple[float, float]] = {
    # Sydney venues  (lat ≈ -33.9)
    "allianz stadium":              (-33.893, 151.225),
    "sydney football stadium":      (-33.893, 151.225),
    "accor stadium":                (-33.847, 151.063),
    "stadium australia":            (-33.847, 151.063),
    "olympic park":                 (-33.847, 151.063),
    "leichhardt oval":              (-33.876, 151.151),
    "commbank stadium":             (-33.814, 151.003),
    "parramatta stadium":           (-33.814, 151.003),
    "bankwest stadium":             (-33.814, 151.003),
    "campbelltown stadium":         (-34.066, 150.814),
    "win stadium":                  (-34.424, 150.903),
    "belmore sports ground":        (-33.916, 151.091),
    "4 pines park":                 (-33.796, 151.282),
    "brookvale oval":               (-33.796, 151.282),
    "manly":                        (-33.796, 151.282),
    "netstrata jubilee stadium":    (-33.970, 151.108),
    "kogarah":                      (-33.970, 151.108),
    "bitv stadium":                 (-33.849, 151.031),
    # Brisbane venues (lat ≈ -27.5)
    "suncorp stadium":              (-27.465, 153.009),
    "lang park":                    (-27.465, 153.009),
    "dolphins/redcliffe":           (-27.236, 153.099),
    "moreton daily stadium":        (-27.236, 153.099),
    # Gold Coast (lat ≈ -28.0)
    "cbus super stadium":           (-27.975, 153.399),
    "robina":                       (-27.975, 153.399),
    # Newcastle (lat ≈ -32.9)
    "mcdonald jones stadium":       (-32.925, 151.729),
    "hunter stadium":               (-32.925, 151.729),
    "newcastle":                    (-32.925, 151.729),
    # Townsville (lat ≈ -19.3)
    "qld country bank stadium":     (-19.317, 146.746),
    "1300smiles stadium":           (-19.317, 146.746),
    "townsville":                   (-19.317, 146.746),
    # Canberra (lat ≈ -35.3)
    "gjamison stadium":             (-35.301, 149.126),
    "gib power stadium":            (-35.301, 149.126),
    "canberra stadium":             (-35.301, 149.126),
    "go-media stadium":             (-35.301, 149.126),
    # Melbourne (lat ≈ -37.8)
    "marvel stadium":               (-37.816, 144.947),
    "etihad stadium":               (-37.816, 144.947),
    "aami park":                    (-37.815, 144.984),
    "melbourne":                    (-37.815, 144.984),
    # Auckland / New Zealand (lat ≈ -36.9)
    "go media stadium":             (-36.920, 174.737),
    "mt smart stadium":             (-36.920, 174.737),
    "eden park":                    (-36.876, 174.743),
    "auckland":                     (-36.920, 174.737),
    # Wollongong (lat ≈ -34.4)
    "win stadium wollongong":       (-34.424, 150.903),
    "wollongong":                   (-34.424, 150.903),
    # Cairns / Darwin (tropical)
    "cazalys stadium":              (-16.920, 145.766),
    "darwin":                       (-12.462, 130.841),
    # Port Moresby / PNG
    "oil search stadium":           (-9.443, 147.180),
    "port moresby":                 (-9.443, 147.180),
}


def _get_venue_lat(venue_name: str) -> float | None:
    """Return latitude for a venue name, trying VENUE_COORDS then fallback."""
    if not isinstance(venue_name, str):
        return None

    vn = venue_name.strip()

    # 1. Try exact match in VENUE_COORDS (Builder 1's module)
    if _HAVE_COORDS and vn in VENUE_COORDS:
        return VENUE_COORDS[vn][0]

    # 2. Try case-insensitive substring match in VENUE_COORDS
    if _HAVE_COORDS:
        vn_lower = vn.lower()
        for k, v in VENUE_COORDS.items():
            if k.lower() in vn_lower or vn_lower in k.lower():
                return v[0]

    # 3. Fallback dict (case-insensitive substring)
    vn_lower = vn.lower()
    for k, coords in _FALLBACK_COORDS.items():
        if k in vn_lower or vn_lower in k:
            return coords[0]

    return None


def compute_weather_proxy_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute weather proxy features from date and venue location.

    Parameters
    ----------
    matches : pd.DataFrame
        Main feature DataFrame.  Must contain:
          - month  : int (1-12), added by v3 engineering features
          - venue  : str, venue name

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 3 new boolean columns appended.
    """
    print("\n" + "=" * 80)
    print("  V4: COMPUTING WEATHER PROXY FEATURES")
    print("=" * 80)

    df = matches.copy()

    # ── Month ─────────────────────────────────────────────────────────────────
    if "month" in df.columns:
        month = pd.to_numeric(df["month"], errors="coerce")
    elif "date" in df.columns:
        month = pd.to_datetime(df["date"], errors="coerce").dt.month
    else:
        month = pd.Series(np.nan, index=df.index)

    # ── Venue latitude ────────────────────────────────────────────────────────
    venue_col = df.get("venue", pd.Series("", index=df.index))
    lat_series = venue_col.map(_get_venue_lat)  # float or None

    # ── is_wet_season ─────────────────────────────────────────────────────────
    # Summer storms: Sydney / Melbourne (lat < -30), Nov–Mar
    wet_months = month.isin([1, 2, 3, 11, 12])
    southern_venue = lat_series.lt(-30)
    df["is_wet_season"] = (wet_months & southern_venue).astype(float)
    df.loc[lat_series.isna(), "is_wet_season"] = np.nan

    # ── is_cold_game ──────────────────────────────────────────────────────────
    # Mid-winter: very southern venues (lat < -33), Jun–Aug
    cold_months = month.isin([6, 7, 8])
    very_southern = lat_series.lt(-33)
    df["is_cold_game"] = (cold_months & very_southern).astype(float)
    df.loc[lat_series.isna(), "is_cold_game"] = np.nan

    # ── is_hot_game ───────────────────────────────────────────────────────────
    # Hot/humid conditions: QLD venues (lat > -28), Oct–Mar (includes season start)
    hot_months = month.isin([10, 11, 12, 1, 2, 3])
    tropical_venue = lat_series.gt(-28)
    df["is_hot_game"] = (hot_months & tropical_venue).astype(float)
    df.loc[lat_series.isna(), "is_hot_game"] = np.nan

    n_new = 3
    known_lat = lat_series.notna().sum()
    total = len(df)
    coverage = lat_series.notna().mean() * 100
    print(f"  Added {n_new} weather proxy features")
    print(f"  Venue lat resolved for {known_lat}/{total} rows ({coverage:.0f}% coverage)")
    print(f"  Wet-season games: {df['is_wet_season'].sum():.0f}, "
          f"Cold: {df['is_cold_game'].sum():.0f}, "
          f"Hot: {df['is_hot_game'].sum():.0f}")

    return df
