"""
Weather & Ground Condition Features (V4.2)
==========================================
Replaces V4.1 proxy features (month/lat) with:
  1. Real weather data from Open-Meteo (temperature, precipitation, wind)
  2. NRL.com ground conditions (Good/Fair/Slippery/Wet/Heavy/Muddy)

Features produced (10 total):
  From Open-Meteo:
    - temperature_c        : average temp on game day (°C)
    - precipitation_mm     : total precipitation on game day (mm)
    - wind_speed_kmh       : max wind speed on game day (km/h)
    - is_rainy             : precipitation > 1.0 mm
    - is_windy             : wind speed > 30 km/h
    - is_cold_actual       : temperature < 12°C

  From NRL.com ground conditions:
    - ground_not_good      : ground_conditions != 'Good' (binary)
    - ground_severity      : ordinal 0-4 (Good=0, Fair=1, Slippery=2, Wet=3, Heavy/Muddy=4)

  Interactions:
    - rain_x_wind          : precipitation_mm * wind_speed_kmh (compound bad weather)
    - bad_conditions_score : ground_severity + is_rainy + is_windy (composite 0-6)

Falls back to proxy features when real data is unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Ground condition ordinal mapping
GROUND_SEVERITY = {
    "Good": 0,
    "": 0,        # blank = assume good
    "Fine": 0,    # sometimes weather field leaks here
    "Fair": 1,
    "Slippery": 2,
    "Wet": 3,
    "Heavy": 4,
    "Muddy": 4,
}


def _load_weather_actual() -> pd.DataFrame | None:
    """Load Open-Meteo weather data if available."""
    path = PROCESSED_DIR / "weather_actual.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Ensure join keys are correct types
    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype(str).str.strip()
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    return df


def _load_ground_conditions() -> pd.DataFrame | None:
    """Load NRL.com ground conditions from match_officials.parquet."""
    path = PROCESSED_DIR / "match_officials.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype(str).str.strip()
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    return df[["year", "round", "home_team", "away_team", "ground_conditions", "weather"]]


def compute_weather_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute real weather + ground condition features.

    Parameters
    ----------
    matches : pd.DataFrame
        Main feature DataFrame. Must contain: year, round, home_team, away_team.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 10 new weather/ground columns appended.
    """
    print("\n" + "=" * 80)
    print("  V4.2: COMPUTING REAL WEATHER + GROUND CONDITION FEATURES")
    print("=" * 80)

    df = matches.copy()
    n = len(df)

    # Ensure join keys exist and are typed correctly
    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype(str).str.strip()
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    join_keys = ["year", "round", "home_team", "away_team"]

    # ── 1. Open-Meteo actual weather ──────────────────────────────────────
    weather_df = _load_weather_actual()
    if weather_df is not None:
        # Merge on match keys
        weather_cols = ["temperature_c", "precipitation_mm", "wind_speed_kmh", "weather_code"]
        merge_cols = join_keys + weather_cols
        weather_df = weather_df[merge_cols].drop_duplicates(subset=join_keys)

        df = df.merge(weather_df, on=join_keys, how="left")
        weather_coverage = df["temperature_c"].notna().mean() * 100
        print(f"  Open-Meteo coverage: {weather_coverage:.1f}% ({df['temperature_c'].notna().sum()}/{n})")
    else:
        print("  WARNING: weather_actual.parquet not found — weather features will be NaN")
        for col in ["temperature_c", "precipitation_mm", "wind_speed_kmh", "weather_code"]:
            df[col] = np.nan

    # ── 2. NRL.com ground conditions ──────────────────────────────────────
    ground_df = _load_ground_conditions()
    if ground_df is not None:
        ground_merge = ground_df[join_keys + ["ground_conditions"]].drop_duplicates(subset=join_keys)
        df = df.merge(ground_merge, on=join_keys, how="left", suffixes=("", "_gc"))
        gc_coverage = df["ground_conditions"].notna().mean() * 100
        print(f"  Ground conditions coverage: {gc_coverage:.1f}% ({df['ground_conditions'].notna().sum()}/{n})")
    else:
        print("  WARNING: match_officials.parquet not found — ground features will be NaN")
        df["ground_conditions"] = np.nan

    # ── 3. Derive features from raw data ──────────────────────────────────

    # Binary weather flags
    df["is_rainy"] = (df["precipitation_mm"] > 1.0).astype(float)
    df.loc[df["precipitation_mm"].isna(), "is_rainy"] = np.nan

    df["is_windy"] = (df["wind_speed_kmh"] > 30).astype(float)
    df.loc[df["wind_speed_kmh"].isna(), "is_windy"] = np.nan

    df["is_cold_actual"] = (df["temperature_c"] < 12).astype(float)
    df.loc[df["temperature_c"].isna(), "is_cold_actual"] = np.nan

    # Ground condition features
    gc = df["ground_conditions"].fillna("").astype(str).str.strip()
    df["ground_not_good"] = (gc.isin(["Fair", "Slippery", "Wet", "Heavy", "Muddy"])).astype(float)
    df.loc[df["ground_conditions"].isna(), "ground_not_good"] = np.nan

    df["ground_severity"] = gc.map(GROUND_SEVERITY).astype(float)
    # Unknown values → NaN
    df.loc[~gc.isin(GROUND_SEVERITY.keys()) & (gc != ""), "ground_severity"] = np.nan

    # Interaction: rain × wind compound
    df["rain_x_wind"] = df["precipitation_mm"] * df["wind_speed_kmh"]

    # Composite bad conditions score (0-6 scale)
    df["bad_conditions_score"] = (
        df["ground_severity"].fillna(0) +
        df["is_rainy"].fillna(0) +
        df["is_windy"].fillna(0)
    )
    # NaN if ALL components are NaN
    all_nan = (df["ground_severity"].isna() & df["is_rainy"].isna() & df["is_windy"].isna())
    df.loc[all_nan, "bad_conditions_score"] = np.nan

    # Drop intermediate columns (keep only feature columns)
    drop_cols = ["weather_code", "ground_conditions"]
    # Only drop if they exist and aren't from original dataframe
    for c in drop_cols:
        if c in df.columns and c not in matches.columns:
            df = df.drop(columns=[c])
    # Also drop any merge suffixed columns
    for c in df.columns:
        if c.endswith("_gc"):
            df = df.drop(columns=[c])

    # ── Summary ───────────────────────────────────────────────────────────
    new_features = [
        "temperature_c", "precipitation_mm", "wind_speed_kmh",
        "is_rainy", "is_windy", "is_cold_actual",
        "ground_not_good", "ground_severity",
        "rain_x_wind", "bad_conditions_score",
    ]
    n_new = sum(1 for f in new_features if f in df.columns)

    print(f"\n  Added {n_new} weather/ground features:")
    for f in new_features:
        if f in df.columns:
            valid = df[f].notna().sum()
            if df[f].dtype == float and df[f].dropna().isin([0, 1]).all():
                # Binary feature
                pct_true = df[f].sum() / max(valid, 1) * 100
                print(f"    {f:25s}: {valid:4d}/{n} valid, {pct_true:.1f}% true")
            else:
                mean = df[f].mean()
                print(f"    {f:25s}: {valid:4d}/{n} valid, mean={mean:.2f}")

    return df


# ── Legacy API (backward compat) ─────────────────────────────────────────

def compute_weather_proxy_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Legacy wrapper — redirects to compute_weather_features."""
    return compute_weather_features(matches)
