"""
Parse ladder (standings) tables from RLP round-ladder HTML pages.

RLP ladder pages use a ``<table class="ladder">`` with two header rows
and data rows with ``class="data"``.  The column layout is:

    Rank | Team | Home(P,W,L,D,F,A,PD) | Away(P,W,L,D,F,A,PD)
    | Overall(P,W,L,D,Bye,F,A,Pts,PD) | FPG | APG

The "Overall" section provides the aggregate stats we primarily care about.
Values of ``"-"`` mean zero.

This module extracts each row into a dict and returns a list ordered by
ladder position.

Public API
----------
- :func:`parse_round_ladder` -- parse a single round-ladder page.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected column names (canonical output keys)
# ---------------------------------------------------------------------------
# These are the Overall stats columns plus home/away breakdowns.
_CANONICAL_COLUMNS: list[str] = [
    "position",
    "team",
    "played",
    "won",
    "lost",
    "drawn",
    "byes",
    "points_for",
    "points_against",
    "points_diff",
    "competition_points",
]

# Extended columns for home/away breakdowns.
_EXTENDED_COLUMNS: list[str] = [
    "home_played", "home_won", "home_lost", "home_drawn",
    "home_for", "home_against", "home_diff",
    "away_played", "away_won", "away_lost", "away_drawn",
    "away_for", "away_against", "away_diff",
    "for_per_game", "against_per_game",
]

# The column order in the RLP ladder table, corresponding to the <td> cells
# in each data row (after the rank and team name cells).
_COLUMN_ORDER: list[str] = [
    # Home stats (7 columns)
    "home_played", "home_won", "home_lost", "home_drawn",
    "home_for", "home_against", "home_diff",
    # Away stats (7 columns)
    "away_played", "away_won", "away_lost", "away_drawn",
    "away_for", "away_against", "away_diff",
    # Overall stats (9 columns)
    "played", "won", "lost", "drawn", "byes",
    "points_for", "points_against", "competition_points", "points_diff",
    # Per-game stats (2 columns)
    "for_per_game", "against_per_game",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(value: str | None) -> Optional[int]:
    """Parse *value* as int, stripping commas/whitespace.

    Returns ``0`` for ``"-"`` and ``None`` on failure.
    """
    if value is None:
        return None
    value = value.strip()
    if value == "-" or value == "":
        return 0
    # Handle signed values like "+25" or "-4"
    value = value.lstrip("+")
    try:
        return int(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _safe_float(value: str | None) -> Optional[float]:
    """Parse *value* as float, returning ``None`` on failure."""
    if value is None:
        return None
    value = value.strip()
    if value == "-" or value == "":
        return None
    try:
        return float(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _extract_team_name(cell: Tag) -> str:
    """Extract the team name from a table cell.

    If the cell contains a link (``<a>``), use its text.  Otherwise use
    the cell's full text content.
    """
    link = cell.find("a")
    if link:
        return link.get_text(strip=True)
    return cell.get_text(strip=True)


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_round_ladder(
    html: str,
    year: Optional[int] = None,
    round_id: Optional[int | str] = None,
) -> list[dict[str, Any]]:
    """Parse all team rows from an RLP round-ladder HTML page.

    Parameters
    ----------
    html:
        Raw HTML content of the round-ladder page.
    year:
        Season year (attached to each row dict for convenience).
    round_id:
        Round identifier (attached to each row dict).

    Returns
    -------
    list[dict]
        One dict per team row, ordered by ladder position (ascending).
        Each dict contains the keys defined in :data:`_CANONICAL_COLUMNS`
        plus extended home/away breakdowns and optional ``year``/``round``
        metadata.

    Notes
    -----
    The RLP ladder table has ``class="ladder"`` and uses two header rows
    (grouping row and individual column headers).  Data rows have
    ``class="data"``.
    """
    soup = BeautifulSoup(html, "lxml")

    # Strategy 1: Find the table with class="ladder".
    ladder_table = soup.find("table", class_="ladder")

    # Strategy 2: Fallback to finding any table with team data.
    if ladder_table is None:
        tables = soup.find_all("table")
        for tbl in tables:
            if tbl.find("td", class_="rank"):
                ladder_table = tbl
                break

    if ladder_table is None:
        logger.warning(
            "No ladder table found on page (year=%s, round=%s).",
            year,
            round_id,
        )
        return []

    # Find all data rows.
    data_rows = ladder_table.find_all("tr", class_=lambda c: c and "data" in c.split())

    if not data_rows:
        # Fallback: try all <tr> that have a <td class="rank">.
        data_rows = [
            tr for tr in ladder_table.find_all("tr")
            if tr.find("td", class_="rank")
        ]

    if not data_rows:
        logger.warning(
            "No data rows found in ladder table (year=%s, round=%s).",
            year,
            round_id,
        )
        return []

    results: list[dict[str, Any]] = []
    position_counter = 0

    for tr in data_rows:
        position_counter += 1
        cells = tr.find_all("td")

        if len(cells) < 3:
            continue

        row: dict[str, Any] = {}

        # First cell: rank (class="rank"), e.g. "1." or "&nbsp;"
        rank_cell = cells[0]
        rank_text = rank_cell.get_text(strip=True).rstrip(".")
        if rank_text and rank_text != "\xa0" and rank_text.strip():
            row["position"] = _safe_int(rank_text)
        else:
            # No explicit rank (happens for tied positions); use counter.
            row["position"] = position_counter

        if row["position"] is None or row["position"] == 0:
            row["position"] = position_counter

        # Second cell: team name (class="name"), contains <a> link.
        name_cell = cells[1]
        row["team"] = _extract_team_name(name_cell)

        if not row["team"]:
            continue

        # Remaining cells: stats in the order defined by _COLUMN_ORDER.
        stat_cells = cells[2:]
        for i, col_name in enumerate(_COLUMN_ORDER):
            if i >= len(stat_cells):
                break
            cell_text = stat_cells[i].get_text(strip=True)

            if col_name in ("for_per_game", "against_per_game"):
                row[col_name] = _safe_float(cell_text)
            elif col_name == "points_diff":
                # Points diff can be "+25", "-4", or "0"
                row[col_name] = _safe_int(cell_text)
            elif col_name in ("home_diff", "away_diff"):
                row[col_name] = _safe_int(cell_text)
            else:
                row[col_name] = _safe_int(cell_text)

        # Fill missing canonical columns with defaults.
        for col in _CANONICAL_COLUMNS:
            row.setdefault(col, None)
        for col in _EXTENDED_COLUMNS:
            row.setdefault(col, None)

        # Compute diff if not provided.
        if row["points_diff"] is None:
            pf = row.get("points_for")
            pa = row.get("points_against")
            if pf is not None and pa is not None:
                row["points_diff"] = pf - pa

        # Attach metadata.
        if year is not None:
            row["year"] = year
        if round_id is not None:
            row["round"] = round_id

        results.append(row)

    # Sort by position just in case the HTML wasn't ordered.
    results.sort(key=lambda r: r.get("position", 999))

    logger.info(
        "Parsed %d teams from ladder (year=%s, round=%s).",
        len(results),
        year,
        round_id,
    )
    return results
