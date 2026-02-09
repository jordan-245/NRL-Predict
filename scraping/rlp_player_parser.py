"""
Parse player data from RLP pages.

Two page types are handled:

1. **Season player list** (``/seasons/nrl-{year}/players.html``):
   A table listing every player who appeared that season.  The RLP table
   uses ``<table class="list x">`` with the following columns:

       Player | Age | Team(s) | Position(s) | APP | INT | TOT | W | L | D
       | W% | T | G | Perc | FG | Pts | (List button)

   Player names are in ``SURNAME, First`` format with an ``<a>`` link.
   Goals (G) may show ``18 / 26`` (made / attempted).

2. **Player profile** (``/players/{slug}/summary.html``):
   Career biography text, positions played, teams represented, and a
   season-by-season stats breakdown table.

Public API
----------
- :func:`parse_season_players` -- parse the season player-list page.
- :func:`parse_player_profile` -- parse an individual player profile page.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Season player list
# ---------------------------------------------------------------------------

# The RLP column order in the season player table.
# These are the canonical column names assigned positionally to the <td> cells.
_PLAYER_COLUMN_ORDER: list[str] = [
    "player",       # Player name (SURNAME, First) with link
    "age",          # Age (int)
    "team",         # Team code(s) e.g. "CBY-14"
    "positions",    # Position codes e.g. "W-14"
    "app_starts",   # Appearances as starter
    "app_int",      # Appearances from interchange
    "app_total",    # Total appearances (APP + INT)
    "wins",         # Wins
    "losses",       # Losses
    "draws",        # Draws
    "win_pct",      # Win percentage (e.g. "57.14%")
    "tries",        # Tries scored
    "goals",        # Goals kicked (may be "18 / 26")
    "goal_pct",     # Goal kicking percentage
    "field_goals",  # Field goals
    "points",       # Total points
    # Last column is a "List" link -- ignored.
]

# Mapping of common header text to canonical column name (for fallback).
_PLAYER_HEADER_MAP: dict[str, str] = {
    "player": "player",
    "name": "player",
    "age": "age",
    "team": "team",
    "teams": "team",
    "team(s)": "team",
    "position": "positions",
    "positions": "positions",
    "position(s)": "positions",
    "app": "app_starts",
    "apps": "app_starts",
    "int": "app_int",
    "tot": "app_total",
    "w": "wins",
    "l": "losses",
    "d": "draws",
    "w%": "win_pct",
    "t": "tries",
    "g": "goals",
    "perc": "goal_pct",
    "fg": "field_goals",
    "pts": "points",
    "points": "points",
}

# Canonical output columns for the season player parser.
_OUTPUT_COLUMNS: list[str] = [
    "player", "player_url", "age", "team", "positions",
    "app_starts", "app_int", "app_total",
    "wins", "losses", "draws", "win_pct",
    "tries", "goals", "goals_attempted", "goal_pct",
    "field_goals", "points",
]

# Legacy output columns kept for backward compatibility.
_LEGACY_COLUMNS: list[str] = [
    "player", "player_url", "team",
    "appearances", "tries", "goals", "field_goals", "points",
]


def _safe_int(value: str | None) -> Optional[int]:
    """Parse *value* as int, stripping commas/whitespace.  None on failure."""
    if value is None:
        return None
    value = value.strip()
    if value == "-" or value == "":
        return 0
    try:
        return int(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _safe_float(value: str | None) -> Optional[float]:
    """Parse *value* as float.  None on failure."""
    if value is None:
        return None
    value = value.strip().rstrip("%")
    if value == "-" or value == "":
        return None
    try:
        return float(value)
    except (ValueError, AttributeError):
        return None


def _extract_text(cell: Tag) -> str:
    """Return the stripped text content of a table cell."""
    return cell.get_text(strip=True)


def _extract_link_text_and_href(cell: Tag) -> tuple[str, Optional[str]]:
    """Return ``(text, href)`` from a cell, preferring an ``<a>`` child."""
    link = cell.find("a")
    if link:
        return link.get_text(strip=True), link.get("href")
    return cell.get_text(strip=True), None


def _parse_goals(text: str) -> tuple[Optional[int], Optional[int]]:
    """Parse goals column which may be "18 / 26" (made/attempted) or just "18".

    Returns (goals_made, goals_attempted).  Either may be None.
    """
    if not text or text.strip() == "-" or text.strip() == "":
        return 0, None

    # Try "N / M" format.
    m = re.match(r"(\d+)\s*/\s*(\d+)", text.strip())
    if m:
        return int(m.group(1)), int(m.group(2))

    # Just a number.
    val = _safe_int(text)
    return val, None


def parse_season_players(
    html: str,
    year: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Parse the season player-list page into a list of player dicts.

    Parameters
    ----------
    html:
        Raw HTML of the ``/seasons/nrl-{year}/players.html`` page.
    year:
        Season year (attached to each row for convenience).

    Returns
    -------
    list[dict]
        One dict per player with keys:

        - ``player`` (str): player name (SURNAME, First)
        - ``player_url`` (str | None): relative URL to the profile page
        - ``age`` (int | None)
        - ``team`` (str): team code(s)
        - ``positions`` (str): position codes
        - ``app_starts`` (int | None): appearances as starter
        - ``app_int`` (int | None): appearances from interchange
        - ``app_total`` (int | None): total appearances
        - ``appearances`` (int | None): alias for app_total (legacy)
        - ``wins`` (int | None)
        - ``losses`` (int | None)
        - ``draws`` (int | None)
        - ``win_pct`` (float | None): win percentage
        - ``tries`` (int | None)
        - ``goals`` (int | None): goals kicked
        - ``goals_attempted`` (int | None): goals attempted
        - ``goal_pct`` (float | None): goal kicking percentage
        - ``field_goals`` (int | None)
        - ``points`` (int | None): total points
        - ``year`` (int | None): the season year, if provided
    """
    soup = BeautifulSoup(html, "lxml")

    # Find the main player table.
    # Strategy 1: table with class="list" (possibly also "x").
    player_table = None
    for tbl in soup.find_all("table", class_="list"):
        # Check if it has a thead with Player column.
        thead = tbl.find("thead")
        if thead:
            header_text = thead.get_text().lower()
            if "player" in header_text or "name" in header_text:
                player_table = tbl
                break

    # Strategy 2: fallback to first table with a <thead>.
    if player_table is None:
        for tbl in soup.find_all("table"):
            thead = tbl.find("thead")
            if thead and "player" in thead.get_text().lower():
                player_table = tbl
                break

    if player_table is None:
        logger.warning("No player table found on season-players page (year=%s).", year)
        return []

    # Extract data rows from <tbody>.
    tbody = player_table.find("tbody")
    if tbody is None:
        tbody = player_table

    data_rows = [tr for tr in tbody.find_all("tr") if tr.find("td")]

    results: list[dict[str, Any]] = []

    for tr in data_rows:
        cells = tr.find_all("td")
        if not cells:
            continue

        row: dict[str, Any] = {"player_url": None, "goals_attempted": None}

        # Parse cells positionally.
        for i, col_name in enumerate(_PLAYER_COLUMN_ORDER):
            if i >= len(cells):
                break

            cell = cells[i]
            cell_text = _extract_text(cell)

            if col_name == "player":
                name, href = _extract_link_text_and_href(cell)
                row["player"] = name
                row["player_url"] = href

            elif col_name == "team":
                row["team"] = cell_text

            elif col_name == "positions":
                row["positions"] = cell_text

            elif col_name == "age":
                row["age"] = _safe_int(cell_text)

            elif col_name == "win_pct":
                row["win_pct"] = _safe_float(cell_text)

            elif col_name == "goals":
                goals_made, goals_attempted = _parse_goals(cell_text)
                row["goals"] = goals_made
                row["goals_attempted"] = goals_attempted

            elif col_name == "goal_pct":
                row["goal_pct"] = _safe_float(cell_text)

            else:
                # Numeric column.
                row[col_name] = _safe_int(cell_text)

        # Skip rows without a player name.
        if not row.get("player"):
            continue

        # Set legacy "appearances" alias.
        row["appearances"] = row.get("app_total")

        # Fill missing columns with None.
        for col in _OUTPUT_COLUMNS:
            row.setdefault(col, None)

        if year is not None:
            row["year"] = year

        results.append(row)

    logger.info("Parsed %d players from season list (year=%s).", len(results), year)
    return results


# ---------------------------------------------------------------------------
# Player profile
# ---------------------------------------------------------------------------

def parse_player_profile(html: str) -> dict[str, Any]:
    """Parse an individual player's profile page.

    Parameters
    ----------
    html:
        Raw HTML of the ``/players/{slug}/summary.html`` page.

    Returns
    -------
    dict
        Player data with keys:

        - ``name`` (str): player's full name
        - ``bio`` (str): free-text biography paragraph(s)
        - ``positions`` (list[str]): positions played
        - ``teams`` (list[str]): teams represented
        - ``career_stats`` (list[dict]): season-by-season rows with keys
          ``season``, ``team``, ``competition``, ``appearances``,
          ``tries``, ``goals``, ``field_goals``, ``points``.
    """
    soup = BeautifulSoup(html, "lxml")
    profile: dict[str, Any] = {
        "name": None,
        "bio": "",
        "positions": [],
        "teams": [],
        "career_stats": [],
    }

    # --- Name -----------------------------------------------------------
    h1 = soup.find("h1")
    if h1:
        profile["name"] = h1.get_text(strip=True)

    # --- Biography / info section ---------------------------------------
    _parse_bio_section(soup, profile)

    # --- Career stats table ---------------------------------------------
    _parse_career_table(soup, profile)

    return profile


def _parse_bio_section(soup: BeautifulSoup, profile: dict[str, Any]) -> None:
    """Extract bio text, positions, and teams from the profile page."""
    content = (
        soup.find("div", class_="content")
        or soup.find("div", id="content")
        or soup.find("main")
        or soup.body
    )
    if content is None:
        return

    # Collect paragraph text for the biography.
    bio_parts: list[str] = []
    for p in content.find_all("p", recursive=False):
        text = p.get_text(strip=True)
        if text:
            bio_parts.append(text)
    profile["bio"] = "\n".join(bio_parts)

    # Positions: look for a line like "Position(s): Fullback, Five-eighth"
    full_text = content.get_text(separator="\n")
    pos_match = re.search(
        r"positions?\s*:\s*(?P<pos>.+?)(?:\n|$)",
        full_text,
        re.IGNORECASE,
    )
    if pos_match:
        raw_positions = pos_match.group("pos")
        profile["positions"] = [
            p.strip() for p in re.split(r"[,/]", raw_positions) if p.strip()
        ]

    # Teams: look for a "Team(s):" or "Club(s):" line.
    teams_match = re.search(
        r"(?:teams?|clubs?)\s*:\s*(?P<teams>.+?)(?:\n|$)",
        full_text,
        re.IGNORECASE,
    )
    if teams_match:
        raw_teams = teams_match.group("teams")
        profile["teams"] = [
            t.strip() for t in re.split(r"[,;]", raw_teams) if t.strip()
        ]

    # Fallback: extract teams from links in an info/details list.
    if not profile["teams"]:
        for li in content.find_all("li"):
            li_text = li.get_text(strip=True).lower()
            if "team" in li_text or "club" in li_text:
                links = li.find_all("a")
                profile["teams"] = [a.get_text(strip=True) for a in links if a.get_text(strip=True)]
                break


# Career stats column mapping.
_CAREER_HEADER_MAP: dict[str, str] = {
    "season": "season",
    "year": "season",
    "team": "team",
    "club": "team",
    "comp": "competition",
    "competition": "competition",
    "app": "appearances",
    "apps": "appearances",
    "appearances": "appearances",
    "games": "appearances",
    "gp": "appearances",
    "t": "tries",
    "tries": "tries",
    "g": "goals",
    "goals": "goals",
    "fg": "field_goals",
    "field goals": "field_goals",
    "pts": "points",
    "points": "points",
    "total": "points",
}


def _parse_career_table(soup: BeautifulSoup, profile: dict[str, Any]) -> None:
    """Extract the season-by-season career stats table."""
    tables = soup.find_all("table")
    if not tables:
        return

    # Find the table whose headers best match career stat columns.
    best_table: Optional[Tag] = None
    best_score = -1
    best_cols: list[str] = []

    for tbl in tables:
        cols = _infer_career_columns(tbl)
        score = sum(1 for c in cols if c in _CAREER_HEADER_MAP.values())
        if score > best_score:
            best_score = score
            best_table = tbl
            best_cols = cols

    if best_table is None or best_score < 2:
        return

    body = best_table.find("tbody") or best_table
    data_rows = [tr for tr in body.find_all("tr") if tr.find("td")]

    for tr in data_rows:
        cells = tr.find_all(["td", "th"])
        row: dict[str, Any] = {}

        for i, cell in enumerate(cells):
            if i >= len(best_cols):
                break
            col = best_cols[i]
            text = cell.get_text(strip=True)

            if col in ("season", "team", "competition"):
                # Extract from link text if available.
                link = cell.find("a")
                row[col] = link.get_text(strip=True) if link else text
            else:
                row[col] = _safe_int(text)

        # Skip totals / summary rows.
        season_val = row.get("season", "")
        if isinstance(season_val, str) and "total" in season_val.lower():
            continue

        # Fill missing columns.
        for col in ("season", "team", "competition", "appearances",
                     "tries", "goals", "field_goals", "points"):
            row.setdefault(col, None)

        profile["career_stats"].append(row)


def _infer_career_columns(table: Tag) -> list[str]:
    """Infer canonical column names for a career stats table."""
    header_cells: list[Tag] = []
    thead = table.find("thead")
    if thead:
        header_cells = thead.find_all("th")
    if not header_cells:
        first_tr = table.find("tr")
        if first_tr:
            header_cells = first_tr.find_all("th")

    if header_cells:
        cols: list[str] = []
        for th in header_cells:
            raw = th.get_text(strip=True).lower().rstrip(".")
            mapped = _CAREER_HEADER_MAP.get(raw)
            cols.append(mapped or raw)
        return cols

    # Fallback positional order.
    return ["season", "team", "competition", "appearances",
            "tries", "goals", "field_goals", "points"]
