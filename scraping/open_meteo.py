"""
Fetch actual historical weather data from Open-Meteo for each NRL match.

Open-Meteo Historical Weather API (free, no key required):
  https://archive-api.open-meteo.com/v1/archive

Creates data/processed/weather_actual.parquet with per-match weather:
  year, round, home_team, away_team, venue,
  temperature_c, precipitation_mm, wind_speed_kmh, weather_code
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config.venues import lookup_venue_coords

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "weather"
OUTPUT_PATH = PROCESSED_DIR / "weather_actual.parquet"

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"

# WMO Weather interpretation codes (used by Open-Meteo)
# 0=Clear, 1-3=Partly cloudy, 45-48=Fog, 51-55=Drizzle,
# 61-65=Rain, 66-67=Freezing rain, 71-77=Snow, 80-82=Showers,
# 85-86=Snow showers, 95-99=Thunderstorm
RAIN_CODES = {51, 53, 55, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}


def _parse_kickoff_hour(kickoff_str: str) -> int | None:
    """Extract hour from kickoff_time string like '7:50 PM' or '19:50'."""
    if not isinstance(kickoff_str, str) or not kickoff_str.strip():
        return None
    s = kickoff_str.strip().upper()
    try:
        # Try "7:50 PM" format
        if "PM" in s or "AM" in s:
            parts = s.replace("AM", "").replace("PM", "").strip().split(":")
            hour = int(parts[0])
            if "PM" in s and hour != 12:
                hour += 12
            elif "AM" in s and hour == 12:
                hour = 0
            return hour
        # Try "19:50" format
        return int(s.split(":")[0])
    except (ValueError, IndexError):
        return None


def fetch_weather_for_venue_dates(
    lat: float, lon: float, dates: list[str], delay: float = 0.15
) -> dict[str, dict]:
    """Fetch daily weather for a venue on specific dates.

    Groups dates into contiguous ranges to minimize API calls.
    Returns {date_str: {temperature_c, precipitation_mm, wind_speed_kmh, weather_code}}.
    """
    if not dates:
        return {}

    sorted_dates = sorted(set(dates))
    results = {}

    # Batch into year chunks (Open-Meteo handles long ranges well)
    date_objs = [pd.Timestamp(d) for d in sorted_dates]
    years = sorted(set(d.year for d in date_objs))

    for year in years:
        year_dates = [d for d in date_objs if d.year == year]
        if not year_dates:
            continue

        start = year_dates[0].strftime("%Y-%m-%d")
        end = year_dates[-1].strftime("%Y-%m-%d")

        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                     "wind_speed_10m_max,weather_code",
            "timezone": "auto",
        }

        try:
            r = requests.get(ARCHIVE_API, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    WARNING: Open-Meteo failed for ({lat},{lon}) {start}→{end}: {e}")
            continue

        daily = data.get("daily", {})
        api_dates = daily.get("time", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        wind = daily.get("wind_speed_10m_max", [])
        wcode = daily.get("weather_code", [])

        # Build lookup by date
        for i, d in enumerate(api_dates):
            if d in [yd.strftime("%Y-%m-%d") for yd in year_dates]:
                t_max = temp_max[i] if i < len(temp_max) else None
                t_min = temp_min[i] if i < len(temp_min) else None
                # Use average of max/min as representative temperature
                if t_max is not None and t_min is not None:
                    temp = round((t_max + t_min) / 2, 1)
                else:
                    temp = t_max or t_min

                results[d] = {
                    "temperature_c": temp,
                    "precipitation_mm": round(precip[i], 1) if i < len(precip) and precip[i] is not None else 0.0,
                    "wind_speed_kmh": round(wind[i], 1) if i < len(wind) and wind[i] is not None else 0.0,
                    "weather_code": int(wcode[i]) if i < len(wcode) and wcode[i] is not None else 0,
                }

        time.sleep(delay)

    return results


def backfill_weather(matches_path: str | Path | None = None, delay: float = 0.15) -> pd.DataFrame:
    """Fetch weather for all historical matches and save to parquet.

    Groups by venue to minimize API calls (~88 venues × ~13 years ≈ ~1000 requests).
    """
    if matches_path is None:
        matches_path = PROCESSED_DIR / "matches.parquet"

    print("\n" + "=" * 60)
    print("  Open-Meteo Weather Backfill")
    print("=" * 60)

    df = pd.read_parquet(matches_path)
    # Need: venue, date (parseable), year, round, home_team, away_team
    df["_date"] = pd.to_datetime(df["parsed_date"], errors="coerce")
    df = df.dropna(subset=["_date"])
    df["_date_str"] = df["_date"].dt.strftime("%Y-%m-%d")

    # Exclude future dates (no archive data)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    df = df[df["_date_str"] <= today].copy()

    # Check cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "weather_all.json"
    cached_results = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cached_results = json.load(f)
        print(f"  Loaded {len(cached_results)} cached venue-date entries")

    # Group by venue
    venue_groups = df.groupby("venue")
    all_rows = []
    venues_fetched = 0
    venues_cached = 0

    for venue, group in venue_groups:
        coords = lookup_venue_coords(venue)
        if coords is None:
            print(f"  SKIP: No GPS for '{venue}' ({len(group)} games)")
            # Still add rows with NaN weather
            for _, row in group.iterrows():
                all_rows.append({
                    "year": row["year"], "round": row["round"],
                    "home_team": row["home_team"], "away_team": row["away_team"],
                    "venue": venue,
                    "temperature_c": np.nan, "precipitation_mm": np.nan,
                    "wind_speed_kmh": np.nan, "weather_code": np.nan,
                })
            continue

        lat, lon = coords
        game_dates = group["_date_str"].tolist()

        # Check which dates need fetching
        cache_key_prefix = f"{round(lat,4)}_{round(lon,4)}"
        dates_to_fetch = []
        for d in game_dates:
            ck = f"{cache_key_prefix}_{d}"
            if ck not in cached_results:
                dates_to_fetch.append(d)

        if dates_to_fetch:
            weather_data = fetch_weather_for_venue_dates(lat, lon, dates_to_fetch, delay=delay)
            for d, vals in weather_data.items():
                ck = f"{cache_key_prefix}_{d}"
                cached_results[ck] = vals
            venues_fetched += 1
        else:
            venues_cached += 1

        # Build output rows
        for _, row in group.iterrows():
            d = row["_date_str"]
            ck = f"{cache_key_prefix}_{d}"
            w = cached_results.get(ck, {})
            all_rows.append({
                "year": row["year"], "round": row["round"],
                "home_team": row["home_team"], "away_team": row["away_team"],
                "venue": venue,
                "temperature_c": w.get("temperature_c", np.nan),
                "precipitation_mm": w.get("precipitation_mm", np.nan),
                "wind_speed_kmh": w.get("wind_speed_kmh", np.nan),
                "weather_code": w.get("weather_code", np.nan),
            })

    # Save cache
    with open(cache_file, "w") as f:
        json.dump(cached_results, f)
    print(f"  Venues fetched: {venues_fetched}, cached: {venues_cached}")

    result = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    # Summary
    n = len(result)
    coverage = result["temperature_c"].notna().mean() * 100
    rainy = (result["precipitation_mm"] > 0.5).sum()
    windy = (result["wind_speed_kmh"] > 30).sum()
    cold = (result["temperature_c"] < 12).sum()
    hot = (result["temperature_c"] > 30).sum()

    print(f"\n  Saved {n} rows to {OUTPUT_PATH}")
    print(f"  Coverage: {coverage:.1f}%")
    print(f"  Rainy games (>0.5mm): {rainy} ({rainy/n*100:.1f}%)")
    print(f"  Windy games (>30km/h): {windy} ({windy/n*100:.1f}%)")
    print(f"  Cold games (<12°C): {cold} ({cold/n*100:.1f}%)")
    print(f"  Hot games (>30°C): {hot} ({hot/n*100:.1f}%)")

    return result


def fetch_upcoming_weather(upcoming_path: str | Path, delay: float = 0.2) -> pd.DataFrame:
    """Fetch forecast weather for upcoming matches (uses Open-Meteo forecast API)."""
    FORECAST_API = "https://api.open-meteo.com/v1/forecast"

    df = pd.read_csv(upcoming_path)
    rows = []

    for _, row in df.iterrows():
        venue = row.get("venue", "")
        date_str = row.get("date", "")
        coords = lookup_venue_coords(venue)

        if coords is None or not date_str:
            rows.append({**row.to_dict(), "temperature_c": np.nan,
                        "precipitation_mm": np.nan, "wind_speed_kmh": np.nan,
                        "weather_code": np.nan})
            continue

        lat, lon = coords
        try:
            d = pd.Timestamp(date_str).strftime("%Y-%m-%d")
            params = {
                "latitude": round(lat, 4), "longitude": round(lon, 4),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                         "wind_speed_10m_max,weather_code",
                "start_date": d, "end_date": d,
                "timezone": "auto",
            }
            r = requests.get(FORECAST_API, params=params, timeout=15)
            r.raise_for_status()
            data = r.json().get("daily", {})

            t_max = (data.get("temperature_2m_max") or [None])[0]
            t_min = (data.get("temperature_2m_min") or [None])[0]
            temp = round((t_max + t_min) / 2, 1) if t_max and t_min else None
            precip = (data.get("precipitation_sum") or [0])[0]
            wind = (data.get("wind_speed_10m_max") or [0])[0]
            wcode = (data.get("weather_code") or [0])[0]

            rows.append({
                **row.to_dict(),
                "temperature_c": temp,
                "precipitation_mm": round(precip, 1) if precip else 0.0,
                "wind_speed_kmh": round(wind, 1) if wind else 0.0,
                "weather_code": int(wcode) if wcode else 0,
            })
        except Exception as e:
            print(f"  WARNING: Forecast failed for {venue} {date_str}: {e}")
            rows.append({**row.to_dict(), "temperature_c": np.nan,
                        "precipitation_mm": np.nan, "wind_speed_kmh": np.nan,
                        "weather_code": np.nan})

        time.sleep(delay)

    return pd.DataFrame(rows)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fetch Open-Meteo weather for NRL matches")
    p.add_argument("--delay", type=float, default=0.15, help="API delay (seconds)")
    p.add_argument("--upcoming", type=str, default=None,
                   help="Path to upcoming round CSV (fetches forecast instead)")
    args = p.parse_args()

    if args.upcoming:
        result = fetch_upcoming_weather(args.upcoming, delay=args.delay)
        print(result[["venue", "temperature_c", "precipitation_mm", "wind_speed_kmh"]].to_string())
    else:
        backfill_weather(delay=args.delay)


if __name__ == "__main__":
    main()
