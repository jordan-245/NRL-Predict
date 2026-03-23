"""
GPS coordinates for NRL venues and team home cities.

Exports
-------
VENUE_COORDS        : dict[str, tuple[float, float]]
    Known NRL venue names (including historic aliases) -> (latitude, longitude).

TEAM_HOME_CITY      : dict[str, str]
    Canonical team name -> home city string.

CITY_COORDS         : dict[str, tuple[float, float]]
    City name -> (latitude, longitude).

haversine_km        : (lat1, lon1, lat2, lon2) -> float
    Great-circle distance in kilometres.

lookup_venue_coords : (venue_name) -> tuple[float, float] | None
    Fuzzy-resolve a venue string to GPS coords (None when unknown).

travel_distance_km  : (team, venue) -> float
    Distance from a team's home city to the given venue (0.0 on any failure).
"""
from __future__ import annotations

import math
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Haversine formula
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two (lat, lon) points."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ---------------------------------------------------------------------------
# Venue coordinates
# Covers current and historical naming-rights names (88+ NRL venues).
# ---------------------------------------------------------------------------

VENUE_COORDS: dict[str, tuple[float, float]] = {

    # ------------------------------------------------------------------ #
    # SYDNEY – Olympic Park / Homebush                                    #
    # ------------------------------------------------------------------ #
    "Accor Stadium":                         (-33.8474, 151.0636),
    "ANZ Stadium":                           (-33.8474, 151.0636),
    "Stadium Australia":                     (-33.8474, 151.0636),
    "Homebush Stadium":                      (-33.8474, 151.0636),
    "Telstra Stadium":                       (-33.8474, 151.0636),
    "Stadium Aust":                          (-33.8474, 151.0636),

    # ------------------------------------------------------------------ #
    # SYDNEY – Sydney Football Stadium (Moore Park)                       #
    # ------------------------------------------------------------------ #
    "Allianz Stadium":                       (-33.8911, 151.2257),
    "Sydney Football Stadium":               (-33.8911, 151.2257),
    "Football Stadium":                      (-33.8911, 151.2257),
    "SFS":                                   (-33.8911, 151.2257),
    "Moore Park Stadium":                    (-33.8911, 151.2257),

    # ------------------------------------------------------------------ #
    # WESTERN SYDNEY – CommBank / Bankwest / Parramatta                   #
    # ------------------------------------------------------------------ #
    "CommBank Stadium":                      (-33.8165, 150.9941),
    "Bankwest Stadium":                      (-33.8165, 150.9941),
    "Western Sydney Stadium":                (-33.8165, 150.9941),
    "Parramatta Stadium":                    (-33.8165, 150.9941),
    "Pirtek Stadium":                        (-33.8165, 150.9941),
    "CUA Stadium":                           (-33.8165, 150.9941),
    "Robbie McCosker Enterprises Stadium":   (-33.8165, 150.9941),

    # ------------------------------------------------------------------ #
    # WESTERN SYDNEY – Penrith                                            #
    # ------------------------------------------------------------------ #
    "BlueBet Stadium":                       (-33.7498, 150.6934),
    "Penrith Stadium":                       (-33.7498, 150.6934),
    "Panthers Stadium":                      (-33.7498, 150.6934),
    "Pepper Stadium":                        (-33.7498, 150.6934),
    "Centrebet Stadium":                     (-33.7498, 150.6934),
    "Carpenters Stadium":                    (-33.7498, 150.6934),

    # ------------------------------------------------------------------ #
    # SYDNEY NORTH – Brookvale / Manly                                    #
    # ------------------------------------------------------------------ #
    "4 Pines Park":                          (-33.7701, 151.2862),
    "Brookvale Oval":                        (-33.7701, 151.2862),
    "Manly-Warringah Football Stadium":      (-33.7701, 151.2862),
    "Lottoland":                             (-33.7701, 151.2862),
    "GIO Stadium Sydney":                    (-33.7701, 151.2862),
    "Manly Warringah Football Stadium":      (-33.7701, 151.2862),

    # ------------------------------------------------------------------ #
    # SYDNEY SOUTH – Cronulla (Shark Park / PointsBet Stadium)            #
    # ------------------------------------------------------------------ #
    "Shark Park":                            (-34.0469, 151.1443),
    "PointsBet Stadium":                     (-34.0469, 151.1443),
    "Pointsbet Stadium":                     (-34.0469, 151.1443),

    # ------------------------------------------------------------------ #
    # SYDNEY INNER – Leichhardt Oval                                      #
    # ------------------------------------------------------------------ #
    "Leichhardt Oval":                       (-33.8682, 151.1550),

    # ------------------------------------------------------------------ #
    # SYDNEY SOUTH-WEST – Campbelltown                                    #
    # ------------------------------------------------------------------ #
    "Campbelltown Stadium":                  (-34.0757, 150.8042),
    "Campbelltown Sports Stadium":           (-34.0757, 150.8042),

    # ------------------------------------------------------------------ #
    # SYDNEY SOUTH – Kogarah / St George (Jubilee Oval)                   #
    # ------------------------------------------------------------------ #
    "Southern Cross Group Stadium":          (-33.9594, 151.1333),
    "Jubilee Oval":                          (-33.9594, 151.1333),
    "Netstrata Jubilee Stadium":             (-33.9594, 151.1333),
    "Netstrata Jubilee Oval":                (-33.9594, 151.1333),
    "Jubilee Stadium":                       (-33.9594, 151.1333),
    "Kogarah Oval":                          (-33.9594, 151.1333),

    # ------------------------------------------------------------------ #
    # SYDNEY WEST – Belmore (Canterbury)                                  #
    # ------------------------------------------------------------------ #
    "Belmore Sports Ground":                 (-33.8987, 151.0921),
    "Belmore Oval":                          (-33.8987, 151.0921),

    # ------------------------------------------------------------------ #
    # CENTRAL COAST – Gosford                                             #
    # ------------------------------------------------------------------ #
    "Industree Group Stadium":               (-33.4383, 151.3422),
    "Central Coast Stadium":                 (-33.4383, 151.3422),
    "Gosford Stadium":                       (-33.4383, 151.3422),
    "Bluetongue Stadium":                    (-33.4383, 151.3422),
    "Totally Workwear Stadium":              (-33.4383, 151.3422),

    # ------------------------------------------------------------------ #
    # NEWCASTLE                                                           #
    # ------------------------------------------------------------------ #
    "McDonald Jones Stadium":                (-32.9188, 151.7563),
    "Hunter Stadium":                        (-32.9188, 151.7563),
    "NIB Stadium Newcastle":                 (-32.9188, 151.7563),
    "Newcastle Stadium":                     (-32.9188, 151.7563),
    "Energy Australia Stadium":              (-32.9188, 151.7563),
    "No.1 Sportsground":                     (-32.9173, 151.7587),
    "No. 1 Sportsground":                    (-32.9173, 151.7587),
    "No 1 Sportsground":                     (-32.9173, 151.7587),
    "International Sportsground":            (-32.9173, 151.7587),

    # ------------------------------------------------------------------ #
    # CANBERRA                                                            #
    # ------------------------------------------------------------------ #
    "GIO Stadium":                           (-35.2831, 149.1259),
    "Canberra Stadium":                      (-35.2831, 149.1259),
    "GIO Canberra Stadium":                  (-35.2831, 149.1259),
    "ANZAC Park":                            (-35.2831, 149.1259),
    "Seiffert Oval":                         (-35.3349, 149.1278),

    # ------------------------------------------------------------------ #
    # WOLLONGONG                                                          #
    # ------------------------------------------------------------------ #
    "WIN Stadium":                           (-34.4316, 150.8939),
    "WIN Entertainment Centre":              (-34.4316, 150.8939),
    "Wollongong Stadium":                    (-34.4316, 150.8939),
    "Industria Stadium":                     (-34.4316, 150.8939),

    # ------------------------------------------------------------------ #
    # BRISBANE                                                            #
    # ------------------------------------------------------------------ #
    "Suncorp Stadium":                       (-27.4648, 153.0095),
    "Lang Park":                             (-27.4648, 153.0095),
    "Queensland Country Bank Suncorp Stadium": (-27.4648, 153.0095),
    "The Gabba":                             (-27.4858, 153.0381),
    "Gabba":                                 (-27.4858, 153.0381),
    "Brisbane Cricket Ground":               (-27.4858, 153.0381),

    # ------------------------------------------------------------------ #
    # REDCLIFFE – Dolphins home                                           #
    # ------------------------------------------------------------------ #
    "Moreton Daily Stadium":                 (-27.2362, 153.1067),
    "Kayo Stadium":                          (-27.2362, 153.1067),
    "Redcliffe Stadium":                     (-27.2362, 153.1067),
    "Dolphin Oval":                          (-27.2362, 153.1067),
    "Kalinga Park":                          (-27.2362, 153.1067),

    # ------------------------------------------------------------------ #
    # GOLD COAST                                                          #
    # ------------------------------------------------------------------ #
    "Cbus Super Stadium":                    (-28.0759, 153.3744),
    "CBUS Super Stadium":                    (-28.0759, 153.3744),
    "Robina Stadium":                        (-28.0759, 153.3744),
    "Skilled Park":                          (-28.0759, 153.3744),
    "Metricon Stadium":                      (-28.0759, 153.3744),
    "CBus Super Stadium":                    (-28.0759, 153.3744),

    # ------------------------------------------------------------------ #
    # SUNSHINE COAST                                                      #
    # ------------------------------------------------------------------ #
    "Sunshine Coast Stadium":               (-26.6891, 153.0664),
    "Bokarina":                              (-26.6891, 153.0664),

    # ------------------------------------------------------------------ #
    # TOWNSVILLE                                                          #
    # ------------------------------------------------------------------ #
    "Queensland Country Bank Stadium":       (-19.2660, 146.8020),
    "1300SMILES Stadium":                    (-19.2660, 146.8020),
    "Townsville Stadium":                    (-19.2660, 146.8020),
    "Dairy Farmers Stadium":                 (-19.2660, 146.8020),
    "Townsville Football Stadium":           (-19.2660, 146.8020),
    "Country Bank Stadium":                  (-19.2660, 146.8020),
    "Townsville Dairy Farmers Stadium":      (-19.2660, 146.8020),

    # ------------------------------------------------------------------ #
    # CAIRNS                                                              #
    # ------------------------------------------------------------------ #
    "Barlow Park":                           (-16.9283, 145.7700),

    # ------------------------------------------------------------------ #
    # TOOWOOMBA                                                           #
    # ------------------------------------------------------------------ #
    "Clive Berghofer Stadium":               (-27.5432, 151.9519),

    # ------------------------------------------------------------------ #
    # MACKAY                                                              #
    # ------------------------------------------------------------------ #
    "BB Print Stadium":                      (-21.1439, 149.1612),
    "Mackay Stadium":                        (-21.1439, 149.1612),

    # ------------------------------------------------------------------ #
    # ROCKHAMPTON                                                         #
    # ------------------------------------------------------------------ #
    "Central Queensland Stadium":            (-23.3809, 150.5071),

    # ------------------------------------------------------------------ #
    # MELBOURNE                                                           #
    # ------------------------------------------------------------------ #
    "AAMI Park":                             (-37.8244, 144.9832),
    "Melbourne Rectangular Stadium":         (-37.8244, 144.9832),
    "Marvel Stadium":                        (-37.8164, 144.9474),
    "Docklands Stadium":                     (-37.8164, 144.9474),
    "Etihad Stadium":                        (-37.8164, 144.9474),
    "Colonial Stadium":                      (-37.8164, 144.9474),
    "MCG":                                   (-37.8200, 144.9831),
    "Melbourne Cricket Ground":              (-37.8200, 144.9831),
    "GMHBA Stadium":                         (-38.1576, 144.3548),  # Geelong

    # ------------------------------------------------------------------ #
    # ADELAIDE                                                            #
    # ------------------------------------------------------------------ #
    "Adelaide Oval":                         (-34.9155, 138.5961),
    "AAMI Stadium":                          (-34.8814, 138.5043),
    "Hindmarsh Stadium":                     (-34.9243, 138.5909),

    # ------------------------------------------------------------------ #
    # PERTH                                                               #
    # ------------------------------------------------------------------ #
    "Optus Stadium":                         (-31.9507, 115.8785),
    "Perth Stadium":                         (-31.9507, 115.8785),
    "Domain Stadium":                        (-31.9509, 115.8402),
    "nib Stadium":                           (-31.9504, 115.8399),
    "Subiaco Oval":                          (-31.9508, 115.8396),
    "HBF Park":                              (-31.9504, 115.8399),

    # ------------------------------------------------------------------ #
    # DARWIN / NT                                                         #
    # ------------------------------------------------------------------ #
    "TIO Stadium":                           (-12.3985, 130.8795),
    "Darwin Stadium":                        (-12.3985, 130.8795),
    "TIO Traeger Park":                      (-23.7041, 133.8823),

    # ------------------------------------------------------------------ #
    # NEW ZEALAND                                                         #
    # ------------------------------------------------------------------ #
    "Go Media Stadium":                      (-37.0050, 174.8614),
    "Mt Smart Stadium":                      (-37.0050, 174.8614),
    "Mount Smart Stadium":                   (-37.0050, 174.8614),
    "Waikato Stadium":                       (-37.7870, 175.2793),
    "FMG Stadium Waikato":                   (-37.7870, 175.2793),
    "Forsyth Barr Stadium":                  (-45.8788, 170.5028),

    # ------------------------------------------------------------------ #
    # UNITED KINGDOM (overseas NRL fixtures)                              #
    # ------------------------------------------------------------------ #
    "Headingley Stadium":                    (53.8196, -1.5839),
    "DW Stadium":                            (53.5417, -2.6333),
    "Totally Wicked Stadium":                (53.4534, -2.7339),
    "St James Park":                         (54.9756, -1.6217),

    # ------------------------------------------------------------------ #
    # PAPUA NEW GUINEA                                                    #
    # ------------------------------------------------------------------ #
    "Oil Search National Football Stadium":  (-9.4747, 147.1467),
    "National Football Stadium":             (-9.4747, 147.1467),
    "PNG Football Stadium":                  (-9.4747, 147.1467),

    # ------------------------------------------------------------------ #
    # REGIONAL NSW                                                        #
    # ------------------------------------------------------------------ #
    "Apex Oval":                             (-32.2569, 148.6011),   # Dubbo
    "Cessnock Sportsground":                 (-32.8328, 151.3570),
    "Carrington Park":                       (-33.7208, 149.5936),   # Bathurst
    "Glen Willow Regional Sports Complex":   (-32.2355, 148.5977),   # Mudgee
    "Scully Park":                           (-30.8763, 150.9293),   # Tamworth
    "Salter Oval":                           (-25.5010, 151.9481),   # Bundaberg
    "C.ex Coffs International Stadium":      (-30.2980, 153.1185),   # Coffs Harbour
    "Lavington Sports Ground":               (-36.0736, 146.9489),   # Albury
}


# ---------------------------------------------------------------------------
# City coordinates (centroids of each team's home city)
# ---------------------------------------------------------------------------

CITY_COORDS: dict[str, tuple[float, float]] = {
    "Sydney":     (-33.8688, 151.2093),
    "Brisbane":   (-27.4698, 153.0251),
    "Melbourne":  (-37.8136, 144.9631),
    "Canberra":   (-35.2835, 149.1281),
    "Newcastle":  (-32.9283, 151.7817),
    "Townsville": (-19.2590, 146.8169),
    "Gold Coast": (-28.0167, 153.4000),
    "Auckland":   (-36.8485, 174.7633),
    "Wollongong": (-34.4278, 150.8936),
}


# ---------------------------------------------------------------------------
# Team home cities (canonical team name -> city)
# ---------------------------------------------------------------------------

TEAM_HOME_CITY: dict[str, str] = {
    "Brisbane Broncos":              "Brisbane",
    "Canberra Raiders":              "Canberra",
    "Canterbury Bulldogs":           "Sydney",
    "Cronulla Sharks":               "Sydney",
    "Dolphins":                      "Brisbane",
    "Gold Coast Titans":             "Gold Coast",
    "Manly Sea Eagles":              "Sydney",
    "Melbourne Storm":               "Melbourne",
    "New Zealand Warriors":          "Auckland",
    "Newcastle Knights":             "Newcastle",
    "North Queensland Cowboys":      "Townsville",
    "Parramatta Eels":               "Sydney",
    "Penrith Panthers":              "Sydney",
    "South Sydney Rabbitohs":        "Sydney",
    "St George Illawarra Dragons":   "Wollongong",
    "Sydney Roosters":               "Sydney",
    "Wests Tigers":                  "Sydney",
}


# ---------------------------------------------------------------------------
# Fuzzy venue lookup — normalise name, try substring matching
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    """Lowercase and remove non-alphanumeric characters for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Pre-build a normalised lookup at import time for fast O(1) exact hits.
_NORMALISED_LOOKUP: dict[str, tuple[float, float]] = {
    _normalise(k): v for k, v in VENUE_COORDS.items()
}

# Sorted list of (normalised_key, coords) pairs for substring fallback.
_SORTED_KEYS: list[tuple[str, tuple[float, float]]] = sorted(
    _NORMALISED_LOOKUP.items(), key=lambda kv: len(kv[0]), reverse=True
)


def lookup_venue_coords(venue: str) -> Optional[tuple[float, float]]:
    """Return (lat, lon) for *venue*, using fuzzy matching when needed.

    Resolution order:
    1. Exact match in ``VENUE_COORDS``
    2. Normalised (case/punctuation-insensitive) exact match
    3. Normalised substring match (input contains known key, or key contains input)

    Returns ``None`` when the venue cannot be resolved.
    """
    if not venue or venue in ("nan", "None", ""):
        return None

    # 1. Exact match
    if venue in VENUE_COORDS:
        return VENUE_COORDS[venue]

    # 2. Normalised exact match
    norm = _normalise(venue)
    if norm in _NORMALISED_LOOKUP:
        return _NORMALISED_LOOKUP[norm]

    # 3. Substring match — prefer the longest matching key to avoid false positives
    for key_norm, coords in _SORTED_KEYS:
        if key_norm and (key_norm in norm or norm in key_norm):
            return coords

    return None


# ---------------------------------------------------------------------------
# Travel distance helper
# ---------------------------------------------------------------------------

def travel_distance_km(team: str, venue: str) -> float:
    """Compute travel distance in km from *team*'s home city to *venue*.

    Returns 0.0 on any failure (unknown team, unknown venue, missing data).
    """
    if not team or not venue:
        return 0.0

    # Try direct lookup; also try stripping whitespace variants
    home_city = TEAM_HOME_CITY.get(team)
    if home_city is None:
        # Partial-match fallback (e.g. "Brisbane" matches "Brisbane Broncos")
        team_lower = team.strip().lower()
        for canonical, city in TEAM_HOME_CITY.items():
            if team_lower in canonical.lower() or canonical.lower() in team_lower:
                home_city = city
                break
    if home_city is None:
        return 0.0

    city_coords = CITY_COORDS.get(home_city)
    if city_coords is None:
        return 0.0

    venue_coords = lookup_venue_coords(str(venue).strip())
    if venue_coords is None:
        return 0.0

    return haversine_km(city_coords[0], city_coords[1], venue_coords[0], venue_coords[1])
