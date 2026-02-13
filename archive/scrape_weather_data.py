"""
Fetch historical weather data and compute travel distance features for NRL matches.

Uses the Open-Meteo Historical Weather API (free, no authentication required)
to retrieve weather conditions at each match venue on match day, and computes
haversine travel distances from each team's home base to the match venue.

Outputs:
    data/processed/weather_data.parquet  -- weather at kickoff for each match
    data/processed/travel_data.parquet   -- travel distance features per match

Usage:
    python scrape_weather_data.py
    python scrape_weather_data.py --skip-weather      # only compute travel
    python scrape_weather_data.py --skip-travel        # only fetch weather
    python scrape_weather_data.py --force-refetch      # ignore weather cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import time
from difflib import get_close_matches
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from config.settings import PROCESSED_DIR, RAW_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEATHER_CACHE_DIR: Path = RAW_DIR / "weather_cache"
WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_RATE_LIMIT = 0.3  # seconds between API requests

EARTH_RADIUS_KM = 6371.0
INTERSTATE_THRESHOLD_KM = 100  # fallback, but we use state groupings primarily

# ---------------------------------------------------------------------------
# Venue coordinate mapping
# ---------------------------------------------------------------------------
# Maps venue names (including historical aliases) to (latitude, longitude).
# Multiple names may map to the same physical venue due to naming rights changes.

VENUE_COORDS: dict[str, tuple[float, float]] = {
    # --- Sydney Olympic Park / Homebush ---
    "Accor Stadium": (-33.847, 151.063),
    "Stadium Australia": (-33.847, 151.063),
    "ANZ Stadium": (-33.847, 151.063),

    # --- Moore Park / Sydney Football Stadium ---
    "Allianz Stadium": (-33.887, 151.222),
    "Sydney Football Stadium": (-33.887, 151.222),
    "SFS": (-33.887, 151.222),
    "Sydney Cricket Ground": (-33.892, 151.225),

    # --- Suncorp Stadium / Lang Park ---
    "Suncorp Stadium": (-27.465, 153.010),
    "Lang Park": (-27.465, 153.010),

    # --- AAMI Park / Melbourne ---
    "AAMI Park": (-37.825, 144.983),
    "Marvel Stadium": (-37.816, 144.948),

    # --- Penrith ---
    "BlueBet Stadium": (-33.751, 150.687),
    "Panthers Stadium": (-33.751, 150.687),
    "Penrith Stadium": (-33.751, 150.687),
    "Pepper Stadium": (-33.751, 150.687),
    "Centrebet Stadium": (-33.751, 150.687),
    "Sportingbet Stadium": (-33.751, 150.687),

    # --- Brookvale ---
    "4 Pines Park": (-33.780, 151.268),
    "Brookvale Oval": (-33.780, 151.268),
    "Lottoland": (-33.780, 151.268),

    # --- Cronulla ---
    "PointsBet Stadium": (-34.040, 151.137),
    "Pointsbet Stadium": (-34.040, 151.137),
    "Shark Park": (-34.040, 151.137),
    "Endeavour Field": (-34.040, 151.137),
    "Remondis Stadium": (-34.040, 151.137),
    "Southern Cross Group Stadium": (-34.040, 151.137),
    "Shark Stadium": (-34.040, 151.137),
    "Sharks Stadium": (-34.040, 151.137),

    # --- Leichhardt ---
    "Leichhardt Oval": (-33.884, 151.156),

    # --- Kogarah / Jubilee ---
    "Netstrata Jubilee Stadium": (-33.959, 151.118),
    "Jubilee Oval": (-33.959, 151.118),
    "Jubilee Stadium": (-33.959, 151.118),
    "Kogarah Oval": (-33.959, 151.118),
    "UOW Jubilee Oval": (-33.959, 151.118),
    "WIN Jubilee Stadium": (-33.959, 151.118),

    # --- Newcastle ---
    "McDonald Jones Stadium": (-32.927, 151.769),
    "Hunter Stadium": (-32.927, 151.769),
    "Newcastle Stadium": (-32.927, 151.769),
    "Industree Group Stadium": (-32.927, 151.769),

    # --- Canberra ---
    "GIO Stadium": (-35.259, 149.097),
    "Canberra Stadium": (-35.259, 149.097),
    "Bruce Stadium": (-35.259, 149.097),

    # --- Auckland ---
    "Mt Smart Stadium": (-36.918, 174.812),
    "Go Media Stadium": (-36.918, 174.812),

    # --- Gold Coast ---
    "Cbus Super Stadium": (-28.073, 153.374),
    "Robina Stadium": (-28.073, 153.374),
    "Skilled Park": (-28.073, 153.374),

    # --- Townsville ---
    "Queensland Country Bank Stadium": (-19.259, 146.798),
    "1300SMILES Stadium": (-19.259, 146.798),
    "1300 Smiles Stadium": (-19.259, 146.798),
    "Townsville Stadium": (-19.259, 146.798),
    "Dairy Farmers Stadium": (-19.259, 146.798),
    "North Queensland Stadium": (-19.259, 146.798),

    # --- Campbelltown ---
    "Campbelltown Stadium": (-34.065, 150.831),
    "Campbelltown Sports Stadium": (-34.065, 150.831),

    # --- Redcliffe / Dolphins ---
    "Kayo Stadium": (-27.229, 153.095),
    "Redcliffe Dolphins Stadium": (-27.229, 153.095),
    "Moreton Daily Stadium": (-27.229, 153.095),

    # --- Las Vegas ---
    "Allegiant Stadium": (36.091, -115.184),

    # --- Parramatta ---
    "CommBank Stadium": (-33.808, 151.005),
    "Bankwest Stadium": (-33.808, 151.005),
    "Western Sydney Stadium": (-33.808, 151.005),
    "Parramatta Stadium": (-33.808, 151.005),
    "Pirtek Stadium": (-33.808, 151.005),

    # --- Wollongong ---
    "WIN Stadium": (-34.440, 150.878),
    "Wollongong Stadium": (-34.440, 150.878),

    # --- Canterbury / Belmore ---
    "Belmore Sports Ground": (-33.918, 151.098),

    # --- Central Coast ---
    "Central Coast Stadium": (-33.434, 151.343),
    "Gosford Stadium": (-33.434, 151.343),
    "Bluetongue Stadium": (-33.434, 151.343),

    # --- Darwin ---
    "TIO Stadium": (-12.432, 130.842),
    "Marrara Oval": (-12.432, 130.842),
    "Darwin": (-12.432, 130.842),

    # --- Mudgee ---
    "Mudgee": (-32.600, 149.592),
    "Glen Willow Stadium": (-32.600, 149.592),
    "Glen Willow Oval": (-32.600, 149.592),

    # --- Bathurst ---
    "Carrington Park": (-33.419, 149.578),
    "Bathurst": (-33.419, 149.578),

    # --- Cairns ---
    "Barlow Park": (-16.920, 145.770),
    "Cairns": (-16.920, 145.770),

    # --- Tamworth ---
    "Scully Park": (-31.083, 148.833),
    "Tamworth": (-31.083, 148.833),

    # --- Sunshine Coast ---
    "Sunshine Coast Stadium": (-26.680, 153.066),

    # --- Dubbo ---
    "Apex Oval": (-32.756, 148.143),
    "Dubbo": (-32.756, 148.143),

    # --- Perth ---
    "Optus Stadium": (-31.951, 115.889),
    "Perth": (-31.951, 115.889),
    "HBF Park": (-31.951, 115.889),
    "nib Stadium": (-31.951, 115.889),

    # --- Adelaide ---
    "Adelaide Oval": (-34.916, 138.596),
    "Adelaide": (-34.916, 138.596),

    # --- New Zealand regional venues ---
    "Eden Park": (-36.875, 174.745),
    "FMG Stadium": (-37.784, 175.316),
    "FMG Stadium Waikato": (-37.784, 175.316),
    "Forsyth Barr Stadium": (-45.879, 170.508),
    "McLean Park": (-39.492, 176.914),
    "Westpac Stadium": (-41.273, 174.785),
    "Sky Stadium": (-41.273, 174.785),
    "Yarrow Stadium": (-39.062, 174.082),
    "Rugby League Park": (-36.906, 174.825),
    "Shaun Johnson Stadium": (-36.906, 174.825),
    "Daniel Anderson Stadium": (-36.906, 174.825),

    # --- Christchurch ---
    "AMI Stadium": (-43.541, 172.534),

    # --- Brisbane Cricket Ground / The Gabba ---
    "Brisbane Cricket Ground": (-27.486, 153.038),

    # --- Regional NSW ---
    "Browne Park": (-23.373, 150.518),  # Rockhampton QLD actually
    "C.ex Coffs International Stadium": (-30.301, 153.113),
    "BCU International Stadium": (-30.301, 153.113),  # Coffs Harbour
    "Lavington Sports Ground": (-36.052, 146.934),  # Albury region
    "Salter Oval": (-24.879, 152.333),  # Bundaberg QLD
    "McDonalds Park": (-36.752, 144.264),  # Bendigo VIC
    "Marley Brown Oval": (-23.853, 151.253),  # Gladstone QLD

    # --- Toowoomba ---
    "Clive Berghofer Stadium": (-27.558, 151.951),

    # --- Gold Coast secondary ---
    "Apollo Projects Stadium": (-28.073, 153.374),  # alternate Gold Coast
    "BB Print Stadium": (-23.348, 150.512),  # Rockhampton QLD
    "polytec Stadium": (-28.073, 153.374),  # Gold Coast area

    # --- Other ---
    "Virgin Australia Stadium": (-28.073, 153.374),  # Gold Coast alternate name
}

# ---------------------------------------------------------------------------
# Team home base coordinates
# ---------------------------------------------------------------------------
TEAM_HOME_COORDS: dict[str, tuple[float, float]] = {
    "Brisbane Broncos": (-27.465, 153.010),
    "North Queensland Cowboys": (-19.259, 146.798),
    "Gold Coast Titans": (-28.073, 153.374),
    "Dolphins": (-27.229, 153.095),
    "Melbourne Storm": (-37.825, 144.983),
    "New Zealand Warriors": (-36.918, 174.812),
    "Penrith Panthers": (-33.751, 150.687),
    "Sydney Roosters": (-33.887, 151.222),
    "South Sydney Rabbitohs": (-33.847, 151.063),
    "Canterbury Bulldogs": (-33.918, 151.098),
    "Manly Sea Eagles": (-33.780, 151.268),
    "Newcastle Knights": (-32.927, 151.769),
    "Canberra Raiders": (-35.259, 149.097),
    "Wests Tigers": (-33.884, 151.156),
    "Cronulla Sharks": (-34.040, 151.137),
    "Parramatta Eels": (-33.808, 151.005),
    "St George Illawarra Dragons": (-34.449, 150.885),
}

# ---------------------------------------------------------------------------
# State/territory groupings for interstate detection
# ---------------------------------------------------------------------------
TEAM_STATES: dict[str, str] = {
    "Penrith Panthers": "NSW",
    "Sydney Roosters": "NSW",
    "South Sydney Rabbitohs": "NSW",
    "Canterbury Bulldogs": "NSW",
    "Manly Sea Eagles": "NSW",
    "Newcastle Knights": "NSW",
    "Cronulla Sharks": "NSW",
    "Parramatta Eels": "NSW",
    "St George Illawarra Dragons": "NSW",
    "Wests Tigers": "NSW",
    "Brisbane Broncos": "QLD",
    "North Queensland Cowboys": "QLD",
    "Gold Coast Titans": "QLD",
    "Dolphins": "QLD",
    "Melbourne Storm": "VIC",
    "Canberra Raiders": "ACT",
    "New Zealand Warriors": "NZ",
}

# Venue state classification based on coordinates / known locations.
# We classify venues by their geographic region for interstate determination.
VENUE_STATES: dict[str, str] = {
    # NSW
    "Accor Stadium": "NSW", "Stadium Australia": "NSW", "ANZ Stadium": "NSW",
    "Allianz Stadium": "NSW", "Sydney Football Stadium": "NSW", "SFS": "NSW",
    "Sydney Cricket Ground": "NSW",
    "BlueBet Stadium": "NSW", "Panthers Stadium": "NSW", "Penrith Stadium": "NSW",
    "Pepper Stadium": "NSW", "Centrebet Stadium": "NSW", "Sportingbet Stadium": "NSW",
    "4 Pines Park": "NSW", "Brookvale Oval": "NSW", "Lottoland": "NSW",
    "PointsBet Stadium": "NSW", "Pointsbet Stadium": "NSW", "Shark Park": "NSW",
    "Endeavour Field": "NSW", "Remondis Stadium": "NSW",
    "Southern Cross Group Stadium": "NSW", "Shark Stadium": "NSW", "Sharks Stadium": "NSW",
    "Leichhardt Oval": "NSW",
    "Netstrata Jubilee Stadium": "NSW", "Jubilee Oval": "NSW", "Jubilee Stadium": "NSW",
    "Kogarah Oval": "NSW", "UOW Jubilee Oval": "NSW", "WIN Jubilee Stadium": "NSW",
    "McDonald Jones Stadium": "NSW", "Hunter Stadium": "NSW", "Newcastle Stadium": "NSW",
    "Industree Group Stadium": "NSW",
    "CommBank Stadium": "NSW", "Bankwest Stadium": "NSW", "Western Sydney Stadium": "NSW",
    "Parramatta Stadium": "NSW", "Pirtek Stadium": "NSW",
    "WIN Stadium": "NSW", "Wollongong Stadium": "NSW",
    "Belmore Sports Ground": "NSW",
    "Central Coast Stadium": "NSW", "Gosford Stadium": "NSW", "Bluetongue Stadium": "NSW",
    "Campbelltown Stadium": "NSW", "Campbelltown Sports Stadium": "NSW",
    "Mudgee": "NSW", "Glen Willow Stadium": "NSW", "Glen Willow Oval": "NSW",
    "Carrington Park": "NSW", "Bathurst": "NSW",
    "Scully Park": "NSW", "Tamworth": "NSW",
    "Apex Oval": "NSW", "Dubbo": "NSW",
    "C.ex Coffs International Stadium": "NSW", "BCU International Stadium": "NSW",
    "Lavington Sports Ground": "NSW",

    # QLD
    "Suncorp Stadium": "QLD", "Lang Park": "QLD",
    "Cbus Super Stadium": "QLD", "Robina Stadium": "QLD", "Skilled Park": "QLD",
    "Queensland Country Bank Stadium": "QLD", "1300SMILES Stadium": "QLD",
    "1300 Smiles Stadium": "QLD", "Townsville Stadium": "QLD",
    "Dairy Farmers Stadium": "QLD", "North Queensland Stadium": "QLD",
    "Kayo Stadium": "QLD", "Redcliffe Dolphins Stadium": "QLD",
    "Moreton Daily Stadium": "QLD",
    "Sunshine Coast Stadium": "QLD",
    "Barlow Park": "QLD", "Cairns": "QLD",
    "Brisbane Cricket Ground": "QLD",
    "Browne Park": "QLD",
    "Salter Oval": "QLD",
    "Marley Brown Oval": "QLD",
    "Clive Berghofer Stadium": "QLD",
    "BB Print Stadium": "QLD",
    "Apollo Projects Stadium": "QLD",
    "polytec Stadium": "QLD",
    "Virgin Australia Stadium": "QLD",

    # VIC
    "AAMI Park": "VIC", "Marvel Stadium": "VIC",
    "McDonalds Park": "VIC",

    # ACT
    "GIO Stadium": "ACT", "Canberra Stadium": "ACT", "Bruce Stadium": "ACT",

    # NT
    "TIO Stadium": "NT", "Marrara Oval": "NT", "Darwin": "NT",

    # NZ
    "Mt Smart Stadium": "NZ", "Go Media Stadium": "NZ",
    "Eden Park": "NZ",
    "FMG Stadium": "NZ", "FMG Stadium Waikato": "NZ",
    "Forsyth Barr Stadium": "NZ",
    "McLean Park": "NZ",
    "Westpac Stadium": "NZ", "Sky Stadium": "NZ",
    "Yarrow Stadium": "NZ",
    "Rugby League Park": "NZ", "Shaun Johnson Stadium": "NZ",
    "Daniel Anderson Stadium": "NZ",
    "AMI Stadium": "NZ",  # Christchurch

    # WA
    "Optus Stadium": "WA", "Perth": "WA", "HBF Park": "WA", "nib Stadium": "WA",

    # SA
    "Adelaide Oval": "SA", "Adelaide": "SA",

    # International
    "Allegiant Stadium": "INTL",
}


# ---------------------------------------------------------------------------
# Utility: haversine distance
# ---------------------------------------------------------------------------

def haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Compute great-circle distance in km between two lat/lon points."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


# ---------------------------------------------------------------------------
# Utility: fuzzy venue lookup
# ---------------------------------------------------------------------------

# Build a case-insensitive lookup for exact matches first.
_VENUE_LOWER_MAP: dict[str, str] = {k.lower(): k for k in VENUE_COORDS}


def resolve_venue_coords(
    venue_name: str,
) -> tuple[float, float] | None:
    """Resolve a venue name to (lat, lon), using exact then fuzzy matching.

    Returns None if no match is found.
    """
    if venue_name in VENUE_COORDS:
        return VENUE_COORDS[venue_name]

    lower = venue_name.lower()
    if lower in _VENUE_LOWER_MAP:
        return VENUE_COORDS[_VENUE_LOWER_MAP[lower]]

    # Fuzzy match against all known venue names.
    candidates = list(VENUE_COORDS.keys())
    matches = get_close_matches(venue_name, candidates, n=1, cutoff=0.65)
    if matches:
        return VENUE_COORDS[matches[0]]

    return None


def resolve_venue_state(venue_name: str) -> str | None:
    """Resolve venue name to its state/territory code."""
    if venue_name in VENUE_STATES:
        return VENUE_STATES[venue_name]

    lower = venue_name.lower()
    for k, v in VENUE_STATES.items():
        if k.lower() == lower:
            return v

    # Fuzzy match
    candidates = list(VENUE_STATES.keys())
    matches = get_close_matches(venue_name, candidates, n=1, cutoff=0.65)
    if matches:
        return VENUE_STATES[matches[0]]

    return None


# ---------------------------------------------------------------------------
# Utility: parse kickoff time
# ---------------------------------------------------------------------------

def parse_kickoff_hour(kickoff_str: str) -> int:
    """Parse a kickoff time string like '8:00 PM.' into a 24-hour integer.

    Returns 19 (7 PM) as default if parsing fails.
    """
    if not kickoff_str or pd.isna(kickoff_str):
        return 19

    cleaned = str(kickoff_str).strip().rstrip(".")
    # Try to extract hour and AM/PM
    match = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)", cleaned, re.IGNORECASE)
    if not match:
        return 19

    hour = int(match.group(1))
    period = match.group(3).upper()

    if period == "PM" and hour != 12:
        hour += 12
    elif period == "AM" and hour == 12:
        hour = 0

    return hour


# ---------------------------------------------------------------------------
# Weather cache
# ---------------------------------------------------------------------------

def _weather_cache_path(lat: float, lon: float, date_str: str) -> Path:
    """Return the cache file path for a given venue+date weather query."""
    # Use a short hash to keep filenames reasonable.
    key = f"{lat:.3f}_{lon:.3f}_{date_str}"
    safe_name = key.replace(".", "p").replace("-", "m")
    return WEATHER_CACHE_DIR / f"{safe_name}.json"


def _read_weather_cache(lat: float, lon: float, date_str: str) -> dict | None:
    """Read cached weather response, or return None."""
    path = _weather_cache_path(lat, lon, date_str)
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _write_weather_cache(lat: float, lon: float, date_str: str, data: dict) -> None:
    """Write weather response to cache."""
    path = _weather_cache_path(lat, lon, date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Open-Meteo API
# ---------------------------------------------------------------------------

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Return a reusable requests session."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "User-Agent": "NRL-Predictor/0.1 (research project; weather data)"
        })
    return _session


def fetch_weather(
    lat: float,
    lon: float,
    date_str: str,
    *,
    force: bool = False,
    max_retries: int = 3,
) -> dict | None:
    """Fetch hourly weather data from Open-Meteo for a single date+location.

    Parameters
    ----------
    lat, lon:
        Venue coordinates.
    date_str:
        Date in YYYY-MM-DD format.
    force:
        If True, bypass cache.
    max_retries:
        Number of retry attempts on failure.

    Returns
    -------
    dict or None
        The parsed JSON response, or None on failure.
    """
    if not force:
        cached = _read_weather_cache(lat, lon, date_str)
        if cached is not None:
            return cached

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": (
            "temperature_2m,apparent_temperature,rain,precipitation,"
            "relative_humidity_2m,wind_speed_10m,wind_gusts_10m,weather_code"
        ),
        "timezone": "Australia/Sydney",
    }

    session = _get_session()
    last_err: BaseException | None = None

    for attempt in range(max_retries):
        try:
            time.sleep(WEATHER_RATE_LIMIT)
            resp = session.get(OPEN_METEO_URL, params=params, timeout=30)

            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Rate limited (429) for %s/%s -- waiting %.0fs",
                    date_str, f"{lat},{lon}", wait,
                )
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                wait = 2 ** attempt
                logger.warning(
                    "Server error %d for %s -- retrying in %.0fs",
                    resp.status_code, date_str, wait,
                )
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                logger.warning(
                    "HTTP %d for weather query %s/%s: %s",
                    resp.status_code, date_str, f"{lat},{lon}",
                    resp.text[:200],
                )
                return None

            data = resp.json()

            # Validate that we got hourly data.
            if "hourly" not in data or "temperature_2m" not in data.get("hourly", {}):
                logger.warning(
                    "Invalid weather response for %s/%s: missing hourly data",
                    date_str, f"{lat},{lon}",
                )
                return None

            _write_weather_cache(lat, lon, date_str, data)
            return data

        except requests.RequestException as exc:
            last_err = exc
            wait = 2 ** attempt
            logger.warning(
                "Request error for %s/%s (attempt %d/%d): %s",
                date_str, f"{lat},{lon}", attempt + 1, max_retries, exc,
            )
            time.sleep(wait)

    logger.error(
        "Failed to fetch weather for %s/%s after %d attempts: %s",
        date_str, f"{lat},{lon}", max_retries, last_err,
    )
    return None


def extract_weather_at_hour(weather_data: dict, hour: int) -> dict:
    """Extract weather variables at a specific hour from the API response.

    Parameters
    ----------
    weather_data:
        Parsed JSON response from Open-Meteo.
    hour:
        Hour of day (0-23) in local timezone.

    Returns
    -------
    dict
        Weather variables at the requested hour.
    """
    hourly = weather_data.get("hourly", {})
    times = hourly.get("time", [])

    # Find the index matching the requested hour.
    idx = None
    for i, t in enumerate(times):
        # times look like "2024-03-07T19:00"
        if f"T{hour:02d}:00" in t:
            idx = i
            break

    if idx is None:
        # Fall back to closest available hour.
        if times:
            idx = min(hour, len(times) - 1)
        else:
            return {
                "temperature_2m": None,
                "apparent_temperature": None,
                "rain": None,
                "precipitation": None,
                "relative_humidity_2m": None,
                "wind_speed_10m": None,
                "wind_gusts_10m": None,
                "weather_code": None,
            }

    fields = [
        "temperature_2m",
        "apparent_temperature",
        "rain",
        "precipitation",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_gusts_10m",
        "weather_code",
    ]
    result = {}
    for f in fields:
        values = hourly.get(f, [])
        result[f] = values[idx] if idx < len(values) else None

    return result


# ---------------------------------------------------------------------------
# Part 1: Fetch weather data for all matches
# ---------------------------------------------------------------------------

def fetch_all_weather(
    matches: pd.DataFrame,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch weather data for every match in the DataFrame.

    Parameters
    ----------
    matches:
        DataFrame with columns: parsed_date, venue, kickoff_time.
    force:
        If True, re-fetch even if cached.

    Returns
    -------
    pd.DataFrame
        One row per match with weather columns.
    """
    n = len(matches)
    logger.info("Fetching weather data for %d matches...", n)

    weather_rows: list[dict] = []
    skipped = 0
    cached_hits = 0
    api_calls = 0
    failed = 0

    for i, (idx, row) in enumerate(matches.iterrows()):
        venue = row.get("venue", "")
        parsed_date = row.get("parsed_date")
        kickoff_time = row.get("kickoff_time", "")

        # Progress reporting.
        if (i + 1) % 100 == 0 or i == 0:
            print(
                f"  Weather: {i + 1}/{n} "
                f"(cached={cached_hits}, api={api_calls}, skipped={skipped}, failed={failed})"
            )

        # Resolve venue coordinates.
        coords = resolve_venue_coords(venue)
        if coords is None:
            logger.warning("Cannot resolve venue '%s' -- skipping weather", venue)
            weather_rows.append({"match_idx": idx})
            skipped += 1
            continue

        lat, lon = coords

        # Parse date.
        if pd.isna(parsed_date):
            logger.warning("Missing date for match index %s -- skipping", idx)
            weather_rows.append({"match_idx": idx})
            skipped += 1
            continue

        date_str = pd.Timestamp(parsed_date).strftime("%Y-%m-%d")
        kickoff_hour = parse_kickoff_hour(kickoff_time)

        # Check cache first.
        cached = _read_weather_cache(lat, lon, date_str)
        if cached is not None and not force:
            cached_hits += 1
            weather_at_kickoff = extract_weather_at_hour(cached, kickoff_hour)
        else:
            weather_data = fetch_weather(lat, lon, date_str, force=force)
            if weather_data is None:
                weather_rows.append({"match_idx": idx})
                failed += 1
                continue
            api_calls += 1
            weather_at_kickoff = extract_weather_at_hour(weather_data, kickoff_hour)

        weather_at_kickoff["match_idx"] = idx
        weather_at_kickoff["venue_lat"] = lat
        weather_at_kickoff["venue_lon"] = lon
        weather_at_kickoff["kickoff_hour"] = kickoff_hour
        weather_at_kickoff["weather_date"] = date_str
        weather_rows.append(weather_at_kickoff)

    print(
        f"  Weather complete: {n} matches, "
        f"cached={cached_hits}, api={api_calls}, skipped={skipped}, failed={failed}"
    )

    weather_df = pd.DataFrame(weather_rows)
    return weather_df


# ---------------------------------------------------------------------------
# Part 2: Compute travel distance features
# ---------------------------------------------------------------------------

def compute_travel_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute travel distance and interstate features for each match.

    Parameters
    ----------
    matches:
        DataFrame with columns: home_team, away_team, venue.

    Returns
    -------
    pd.DataFrame
        One row per match with travel columns.
    """
    n = len(matches)
    logger.info("Computing travel features for %d matches...", n)

    travel_rows: list[dict] = []
    venue_miss = 0
    team_miss = 0

    for i, (idx, row) in enumerate(matches.iterrows()):
        venue = row.get("venue", "")
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")

        venue_coords = resolve_venue_coords(venue)

        if venue_coords is None:
            logger.warning("Cannot resolve venue '%s' for travel calc", venue)
            travel_rows.append({
                "match_idx": idx,
                "home_travel_km": None,
                "away_travel_km": None,
                "travel_diff_km": None,
                "home_is_interstate": None,
                "away_is_interstate": None,
            })
            venue_miss += 1
            continue

        venue_lat, venue_lon = venue_coords
        venue_state = resolve_venue_state(venue)

        # Home team travel.
        home_coords = TEAM_HOME_COORDS.get(home_team)
        if home_coords:
            home_travel = haversine_km(
                home_coords[0], home_coords[1], venue_lat, venue_lon
            )
        else:
            logger.warning("Unknown home team '%s' for travel calc", home_team)
            home_travel = None
            team_miss += 1

        # Away team travel.
        away_coords = TEAM_HOME_COORDS.get(away_team)
        if away_coords:
            away_travel = haversine_km(
                away_coords[0], away_coords[1], venue_lat, venue_lon
            )
        else:
            logger.warning("Unknown away team '%s' for travel calc", away_team)
            away_travel = None
            team_miss += 1

        # Travel difference (positive = away team travels further).
        if home_travel is not None and away_travel is not None:
            travel_diff = away_travel - home_travel
        else:
            travel_diff = None

        # Interstate detection based on state groupings.
        home_state = TEAM_STATES.get(home_team)
        away_state = TEAM_STATES.get(away_team)

        if home_state and venue_state:
            home_is_interstate = home_state != venue_state
        else:
            home_is_interstate = None

        if away_state and venue_state:
            away_is_interstate = away_state != venue_state
        else:
            away_is_interstate = None

        travel_rows.append({
            "match_idx": idx,
            "home_travel_km": round(home_travel, 1) if home_travel is not None else None,
            "away_travel_km": round(away_travel, 1) if away_travel is not None else None,
            "travel_diff_km": round(travel_diff, 1) if travel_diff is not None else None,
            "home_is_interstate": home_is_interstate,
            "away_is_interstate": away_is_interstate,
        })

    if venue_miss:
        print(f"  Travel: {venue_miss} matches had unresolvable venues")
    if team_miss:
        print(f"  Travel: {team_miss} team lookups failed")

    travel_df = pd.DataFrame(travel_rows)
    return travel_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(
    *,
    skip_weather: bool = False,
    skip_travel: bool = False,
    force_refetch: bool = False,
) -> None:
    """Run the weather + travel data pipeline."""
    t_start = time.time()

    print("=" * 70)
    print("  NRL Weather & Travel Data Pipeline")
    print(f"  Output: {PROCESSED_DIR}")
    print(f"  Cache:  {WEATHER_CACHE_DIR}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load matches
    # ------------------------------------------------------------------
    matches_path = PROCESSED_DIR / "matches.parquet"
    if not matches_path.exists():
        print(f"ERROR: Matches file not found: {matches_path}")
        print("  Run 'python run_scrape.py' first to generate matches.parquet")
        sys.exit(1)

    matches = pd.read_parquet(matches_path)
    print(f"\nLoaded {len(matches)} matches from {matches_path}")
    print(f"  Years: {matches['year'].min()}-{matches['year'].max()}")
    print(f"  Venues: {matches['venue'].nunique()} unique")

    # Verify venue coverage.
    unique_venues = matches["venue"].dropna().unique()
    unmapped = [v for v in unique_venues if resolve_venue_coords(v) is None]
    if unmapped:
        print(f"\n  WARNING: {len(unmapped)} venues could not be mapped:")
        for v in sorted(unmapped):
            count = len(matches[matches["venue"] == v])
            print(f"    - '{v}' ({count} matches)")
    else:
        print(f"\n  All {len(unique_venues)} venues successfully mapped to coordinates.")

    # ------------------------------------------------------------------
    # Part 1: Weather data
    # ------------------------------------------------------------------
    if not skip_weather:
        print(f"\n{'=' * 60}")
        print("  Part 1: Fetching Weather Data (Open-Meteo API)")
        print(f"{'=' * 60}")

        weather_df = fetch_all_weather(matches, force=force_refetch)

        # Merge match identifiers for the output.
        id_cols = ["year", "round", "home_team", "away_team", "parsed_date", "venue"]
        id_cols_available = [c for c in id_cols if c in matches.columns]
        match_ids = matches[id_cols_available].copy()
        match_ids["match_idx"] = matches.index

        weather_out = match_ids.merge(weather_df, on="match_idx", how="left")
        weather_out = weather_out.drop(columns=["match_idx"])

        # Save.
        weather_path = PROCESSED_DIR / "weather_data.parquet"
        weather_out.to_parquet(weather_path, index=False, engine="pyarrow")
        print(f"\n  Saved weather data -> {weather_path}")
        print(f"  Shape: {weather_out.shape}")

        # Summary statistics.
        weather_fields = [
            "temperature_2m", "apparent_temperature", "rain", "precipitation",
            "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m", "weather_code",
        ]
        available_fields = [f for f in weather_fields if f in weather_out.columns]
        if available_fields:
            print("\n  Weather data summary:")
            non_null_counts = weather_out[available_fields].notna().sum()
            for f in available_fields:
                pct = non_null_counts[f] / len(weather_out) * 100
                if weather_out[f].notna().any():
                    mean_val = weather_out[f].dropna().mean()
                    print(f"    {f}: {non_null_counts[f]}/{len(weather_out)} ({pct:.1f}%) -- mean={mean_val:.1f}")
                else:
                    print(f"    {f}: {non_null_counts[f]}/{len(weather_out)} ({pct:.1f}%)")
    else:
        print("\n  Skipping weather data (--skip-weather)")

    # ------------------------------------------------------------------
    # Part 2: Travel distances
    # ------------------------------------------------------------------
    if not skip_travel:
        print(f"\n{'=' * 60}")
        print("  Part 2: Computing Travel Distance Features")
        print(f"{'=' * 60}")

        travel_df = compute_travel_features(matches)

        # Merge match identifiers for the output.
        id_cols = ["year", "round", "home_team", "away_team", "parsed_date", "venue"]
        id_cols_available = [c for c in id_cols if c in matches.columns]
        match_ids = matches[id_cols_available].copy()
        match_ids["match_idx"] = matches.index

        travel_out = match_ids.merge(travel_df, on="match_idx", how="left")
        travel_out = travel_out.drop(columns=["match_idx"])

        # Save.
        travel_path = PROCESSED_DIR / "travel_data.parquet"
        travel_out.to_parquet(travel_path, index=False, engine="pyarrow")
        print(f"\n  Saved travel data -> {travel_path}")
        print(f"  Shape: {travel_out.shape}")

        # Summary statistics.
        travel_cols = ["home_travel_km", "away_travel_km", "travel_diff_km"]
        available_travel = [c for c in travel_cols if c in travel_out.columns]
        if available_travel:
            print("\n  Travel data summary:")
            for col in available_travel:
                vals = travel_out[col].dropna()
                if len(vals) > 0:
                    print(
                        f"    {col}: "
                        f"mean={vals.mean():.1f} km, "
                        f"median={vals.median():.1f} km, "
                        f"max={vals.max():.1f} km, "
                        f"non-null={len(vals)}/{len(travel_out)}"
                    )

        # Interstate stats.
        for side in ["home", "away"]:
            col = f"{side}_is_interstate"
            if col in travel_out.columns:
                interstate = travel_out[col].dropna()
                if len(interstate) > 0:
                    n_interstate = interstate.sum()
                    pct = n_interstate / len(interstate) * 100
                    print(f"    {side} interstate: {int(n_interstate)}/{len(interstate)} ({pct:.1f}%)")
    else:
        print("\n  Skipping travel data (--skip-travel)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Pipeline completed in {elapsed:.1f}s")
    print(f"  Output directory: {PROCESSED_DIR}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch weather data and compute travel features for NRL matches.",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip weather data fetching (only compute travel features).",
    )
    parser.add_argument(
        "--skip-travel",
        action="store_true",
        help="Skip travel distance computation (only fetch weather data).",
    )
    parser.add_argument(
        "--force-refetch",
        action="store_true",
        help="Force re-fetching weather data even if cached.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        skip_weather=args.skip_weather,
        skip_travel=args.skip_travel,
        force_refetch=args.force_refetch,
    )
