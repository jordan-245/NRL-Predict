"""
Team name standardisation for the NRL Match Winner Prediction project.

Every data source (Rugby League Project, AusSportsBetting, NRL official site)
uses slightly different team names.  This module provides:

* ``TEAM_ALIASES``  -- canonical name -> list of known aliases
* ``TEAM_SLUGS``    -- canonical name -> URL-safe slug (for RLP URLs)
* ``standardise_team_name(name)`` -- map *any* alias to its canonical form
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical team names -> known aliases
# ---------------------------------------------------------------------------
# The dictionary key is the **canonical** name used throughout the project.
# The alias list should include every spelling, abbreviation, or legacy name
# that might appear in any data source.  The canonical name itself is always
# implicitly included (see ``_build_lookup``).

TEAM_ALIASES: dict[str, list[str]] = {
    "Penrith Panthers": [
        "Penrith",
        "Panthers",
        "PEN",
    ],
    "Melbourne Storm": [
        "Melbourne",
        "Storm",
        "MEL",
    ],
    "Sydney Roosters": [
        "Sydney",
        "Roosters",
        "Eastern Suburbs",
        "Eastern Suburbs Roosters",
        "Easts",
        "SYD",
    ],
    "South Sydney Rabbitohs": [
        "South Sydney",
        "Souths",
        "Rabbitohs",
        "SSR",
        "SOU",
    ],
    "Canterbury Bulldogs": [
        "Canterbury",
        "Canterbury-Bankstown Bulldogs",
        "Canterbury Bankstown Bulldogs",
        "Canterbury-Bankstown",
        "Bulldogs",
        "CBY",
        "CAN",
    ],
    "Cronulla Sharks": [
        "Cronulla",
        "Cronulla-Sutherland Sharks",
        "Cronulla Sutherland Sharks",
        "Cronulla-Sutherland",
        "Sharks",
        "CRO",
    ],
    "St George Illawarra Dragons": [
        "St George Illawarra",
        "St Geo Illa",
        "St. George Illawarra",
        "St. George Illawarra Dragons",
        "St George Dragons",
        "Dragons",
        "SGI",
    ],
    "Manly Sea Eagles": [
        "Manly",
        "Manly-Warringah Sea Eagles",
        "Manly Warringah Sea Eagles",
        "Manly-Warringah",
        "Sea Eagles",
        "MAN",
    ],
    "New Zealand Warriors": [
        "New Zealand",
        "NZ Warriors",
        "Warriors",
        "NZW",
        "WAR",
    ],
    "North Queensland Cowboys": [
        "North Queensland",
        "North Qld",
        "North QLD Cowboys",
        "North Qld Cowboys",
        "Cowboys",
        "NQL",
        "NQC",
    ],
    "Parramatta Eels": [
        "Parramatta",
        "Eels",
        "PAR",
    ],
    "Brisbane Broncos": [
        "Brisbane",
        "Broncos",
        "BRI",
    ],
    "Newcastle Knights": [
        "Newcastle",
        "Knights",
        "NEW",
        "NCL",
    ],
    "Canberra Raiders": [
        "Canberra",
        "Raiders",
        "CAN Raiders",
        "CBR",
    ],
    "Wests Tigers": [
        "West Tigers",
        "Tigers",
        "Wests",
        "WST",
    ],
    "Gold Coast Titans": [
        "Gold Coast",
        "Titans",
        "GCT",
        "GLD",
    ],
    "Dolphins": [
        "Redcliffe Dolphins",
        "Redcliffe",
        "DOL",
    ],
}

# ---------------------------------------------------------------------------
# Canonical team name -> RLP URL slug
# ---------------------------------------------------------------------------
# Used when constructing match URLs on rugbyleagueproject.org.
# e.g.  "melbourne-storm-vs-penrith-panthers"

TEAM_SLUGS: dict[str, str] = {
    "Penrith Panthers": "penrith-panthers",
    "Melbourne Storm": "melbourne-storm",
    "Sydney Roosters": "sydney-roosters",
    "South Sydney Rabbitohs": "south-sydney-rabbitohs",
    "Canterbury Bulldogs": "canterbury-bankstown-bulldogs",
    "Cronulla Sharks": "cronulla-sutherland-sharks",
    "St George Illawarra Dragons": "st-george-illawarra-dragons",
    "Manly Sea Eagles": "manly-warringah-sea-eagles",
    "New Zealand Warriors": "new-zealand-warriors",
    "North Queensland Cowboys": "north-queensland-cowboys",
    "Parramatta Eels": "parramatta-eels",
    "Brisbane Broncos": "brisbane-broncos",
    "Newcastle Knights": "newcastle-knights",
    "Canberra Raiders": "canberra-raiders",
    "Wests Tigers": "wests-tigers",
    "Gold Coast Titans": "gold-coast-titans",
    "Dolphins": "dolphins",
}

# ---------------------------------------------------------------------------
# Reverse lookup: alias (lowercased) -> canonical name
# ---------------------------------------------------------------------------

def _build_lookup() -> dict[str, str]:
    """Build a case-insensitive alias -> canonical name mapping.

    Includes every explicit alias **and** the canonical name itself so that
    ``standardise_team_name("Melbourne Storm")`` works without special-casing.
    """
    lookup: dict[str, str] = {}
    for canonical, aliases in TEAM_ALIASES.items():
        # Include the canonical name as an alias of itself.
        for alias in [canonical, *aliases]:
            key = alias.strip().lower()
            if key in lookup and lookup[key] != canonical:
                raise ValueError(
                    f"Duplicate alias '{alias}' maps to both "
                    f"'{lookup[key]}' and '{canonical}'. "
                    f"Remove the ambiguous alias from one entry."
                )
            lookup[key] = canonical
    return lookup


_ALIAS_LOOKUP: dict[str, str] = _build_lookup()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def standardise_team_name(name: str) -> str:
    """Return the canonical team name for *name*.

    Matching is **case-insensitive** and strips leading/trailing whitespace.

    Parameters
    ----------
    name:
        Any known team name, abbreviation, or alias.

    Returns
    -------
    str
        The canonical team name (e.g. ``"Melbourne Storm"``).

    Raises
    ------
    KeyError
        If *name* does not match any known alias.

    Examples
    --------
    >>> standardise_team_name("Souths")
    'South Sydney Rabbitohs'
    >>> standardise_team_name("cronulla-sutherland sharks")
    'Cronulla Sharks'
    >>> standardise_team_name("MEL")
    'Melbourne Storm'
    """
    key = name.strip().lower()
    try:
        return _ALIAS_LOOKUP[key]
    except KeyError:
        raise KeyError(
            f"Unknown team name '{name}'. "
            f"Add it to TEAM_ALIASES in config/team_mappings.py."
        ) from None


def get_team_slug(name: str) -> str:
    """Return the RLP URL slug for a team, accepting any known alias.

    Parameters
    ----------
    name:
        Any known team name or alias.

    Returns
    -------
    str
        The URL slug, e.g. ``"melbourne-storm"``.

    Raises
    ------
    KeyError
        If *name* cannot be resolved to a canonical team, or the canonical
        team has no slug entry.
    """
    canonical = standardise_team_name(name)
    try:
        return TEAM_SLUGS[canonical]
    except KeyError:
        raise KeyError(
            f"No URL slug defined for '{canonical}'. "
            f"Add it to TEAM_SLUGS in config/team_mappings.py."
        ) from None


# Convenience: sorted list of all 17 canonical team names.
ALL_TEAMS: list[str] = sorted(TEAM_ALIASES.keys())
