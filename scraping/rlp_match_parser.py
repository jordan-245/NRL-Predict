"""
Parse match details from RLP round-summary HTML pages.

RLP round-summary pages list all matches for a given round inside styled
``<div>`` blocks.  Each match block has this real-world HTML structure::

    <div class="quiet" style="font-size:130%;...;border-bottom:dotted 1px #ccc;">
      <span class="noprint"><a class="rlplnk" href="/matches/62004">></a></span>
      <strong><a href="...">Home</a> 36</strong>
      (scorer links tries; kicker N goals) defeated
      <strong><a href="...">Away</a> 24</strong>
      (scorer links tries; kicker N goals)
      at <a href="...">Venue</a>.<br/>
      Date: Sat, 2nd March. ... Halftime: Home 12-10. ...
      Referee: <a>Name</a>. Crowd: 40,746.
      <div class="quiet small">
        <strong><a>Home</a>:</strong>
        player links, ...; player links; ... <em>Int:</em> bench links.
      </div>
      <div class="quiet small">
        <strong><a>Away</a>:</strong>
        player links, ...; ...
      </div>
    </div>

This module uses BeautifulSoup to locate match blocks and then extracts
every available datum via structured HTML parsing and regex.

Public API
----------
- :func:`parse_round_summary` -- parse an entire round-summary page.
- :func:`parse_match_block`   -- parse a single match block element.

Both return lists of dicts with a consistent schema (see
:data:`MATCH_FIELDS`).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema reference (output dict keys)
# ---------------------------------------------------------------------------
MATCH_FIELDS: list[str] = [
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "result_text",         # e.g. "defeated", "drew with", "lost to"
    "home_try_scorers",    # list[dict] with {player, count}
    "away_try_scorers",
    "home_goal_kickers",   # list[dict] with {player, count}
    "away_goal_kickers",
    "home_field_goals",    # list[dict] with {player, count}
    "away_field_goals",
    "venue",
    "date",                # str (raw date string)
    "parsed_date",         # datetime | None
    "kickoff_time",
    "halftime_home",
    "halftime_away",
    "penalty_home",
    "penalty_away",
    "referee",
    "attendance",          # int | None
    "home_lineup",         # list of player names (positions 1-13)
    "home_bench",          # list of player names (interchange)
    "away_lineup",
    "away_bench",
    "is_bye",
    "is_walkover",
    "is_abandoned",
    "match_url",           # link to the individual match page, if present
]


# ---------------------------------------------------------------------------
# Regex patterns for the metadata line
# ---------------------------------------------------------------------------

# Date:  "Date: Sat, 2nd March."  or  "Date: Thu, 7th March."
# Note: RLP omits the year on the date line; the year is inferred from context.
_DATE_RE = re.compile(
    r"Date:\s*(?P<day>\w+),?\s*(?P<date>\d{1,2}\w*\s+\w+)",
    re.IGNORECASE,
)

# Kickoff: "Kickoff: 8:00 PM." or "Kickoff: 6:30 PM."
_KICKOFF_RE = re.compile(
    r"Kickoff:\s*(?P<time>[\d:]+\s*[APap]\.?[Mm]\.?)",
    re.IGNORECASE,
)

# Halftime: "Halftime: Manly\t12-10." or "Halftime: Canberra\t\t8-6."
# or "Halftime: 12-10." (no team prefix)
_HALFTIME_RE = re.compile(
    r"Halftime:\s*(?:[\w\s.'-]+?)?\s*(?P<home>\d+)\s*-\s*(?P<away>\d+)",
    re.IGNORECASE,
)

# Penalties: "Penalties: Manly 5-1." or "Penalties: 6-all." or "Penalties: 8-all."
_PENALTIES_RE = re.compile(
    r"Penalties:\s*(?:[\w\s.'-]+?)?\s*(?P<home>\d+)\s*-\s*(?P<away>\d+|all)",
    re.IGNORECASE,
)

# Crowd: "Crowd: 40,746."
_CROWD_RE = re.compile(
    r"Crowd:\s*(?P<crowd>[\d,]+)",
    re.IGNORECASE,
)

# Result verb between the two team score blocks in the text representation.
_RESULT_VERB_RE = re.compile(
    r"\b(defeated|drew\s+with|lost\s+to|beat)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Scorer parsing helpers
# ---------------------------------------------------------------------------

def _parse_scorers_from_text(text: str) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Parse an inline scorer parenthetical into tries, goals, field goals.

    The text is the content inside parentheses after a team score, e.g.::

        "R. Smith try; N. Meaney 2 goals"
        "L. Brooks, L. Croker, R. Garrick tries; R. Garrick 6 goals"
        "T. Dearden tries; V. Holmes 7 goals; C. Townsend field goal"

    Returns
    -------
    tuple of (try_scorers, goal_kickers, field_goals)
        Each is a list of ``{player: str, count: int}`` dicts.
    """
    tries: list[dict[str, Any]] = []
    goals: list[dict[str, Any]] = []
    field_goals: list[dict[str, Any]] = []

    if not text or not text.strip():
        return tries, goals, field_goals

    # Split on semicolons to separate tries, goals, and field goals sections.
    sections = text.split(";")
    for section in sections:
        section = section.strip()
        if not section:
            continue

        section_lower = section.lower()

        if "field goal" in section_lower:
            # Remove the label: "field goal" or "field goals"
            cleaned = re.sub(r"\s*field\s+(?:goals|goal)\s*$", "", section, flags=re.IGNORECASE).strip()
            fg_list = _parse_scorer_list(cleaned)
            field_goals.extend(fg_list)
        elif "goal" in section_lower:
            # Remove the label: "goal" or "goals"
            cleaned = re.sub(r"\s*(?:goals|goal)\s*$", "", section, flags=re.IGNORECASE).strip()
            goal_list = _parse_goal_list(cleaned)
            goals.extend(goal_list)
        elif "try" in section_lower or "tries" in section_lower:
            # Remove the label: "try" or "tries"
            cleaned = re.sub(r"\s*(?:tries|try)\s*$", "", section, flags=re.IGNORECASE).strip()
            try_list = _parse_scorer_list(cleaned)
            tries.extend(try_list)
        else:
            # Fallback: assume tries if no label present
            scorer_list = _parse_scorer_list(section)
            tries.extend(scorer_list)

    return tries, goals, field_goals


def _parse_scorer_list(text: str) -> list[dict[str, Any]]:
    """Parse a comma-separated list of scorers (tries or field goals).

    Handles: "J. Smith 2, B. Jones, T. Brown 3" and "J. Smith"
    The count appears after the last player name if > 1.
    """
    scorers: list[dict[str, Any]] = []
    if not text or not text.strip():
        return scorers

    for segment in text.split(","):
        segment = segment.strip()
        if not segment:
            continue
        # Match "Name N" pattern where N is the count.
        m = re.match(
            r"(?P<name>.+?)\s+(?P<count>\d+)\s*$",
            segment,
        )
        if m:
            name = m.group("name").strip()
            if name and re.match(r"[A-Za-z]", name):
                scorers.append({
                    "player": name,
                    "count": int(m.group("count")),
                })
        else:
            # Name only, count=1.
            name = segment.strip().rstrip(".")
            if name and re.match(r"[A-Za-z]", name):
                scorers.append({"player": name, "count": 1})
    return scorers


def _parse_goal_list(text: str) -> list[dict[str, Any]]:
    """Parse goal kickers from text like "R. Garrick 6" or "N. Hynes, B. Trindall".

    Goal entries can be "Name N" (N = number of goals) or just "Name" (1 goal).
    """
    kickers: list[dict[str, Any]] = []
    if not text or not text.strip():
        return kickers

    for segment in text.split(","):
        segment = segment.strip()
        if not segment:
            continue
        # Match "Name N" where N is goal count.
        m = re.match(
            r"(?P<name>.+?)\s+(?P<count>\d+)\s*$",
            segment,
        )
        if m:
            name = m.group("name").strip()
            if name and re.match(r"[A-Za-z]", name):
                kickers.append({
                    "player": name,
                    "count": int(m.group("count")),
                })
        else:
            name = segment.strip().rstrip(".")
            if name and re.match(r"[A-Za-z]", name):
                kickers.append({"player": name, "count": 1})
    return kickers


def _safe_int(value: str | None) -> Optional[int]:
    """Convert a string to int, stripping commas.  Return None on failure."""
    if value is None:
        return None
    try:
        return int(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _parse_date(raw: str, year: Optional[int] = None) -> Optional[datetime]:
    """Attempt to parse a raw date string into a datetime object.

    RLP date lines are like "2nd March" (no year) or "8 March 2024".
    If no year in the string, *year* is appended.
    """
    # Strip ordinal suffixes: "2nd" -> "2", "7th" -> "7", "1st" -> "1"
    cleaned = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", raw.strip())

    # If the string doesn't contain a 4-digit year, append one.
    if year and not re.search(r"\d{4}", cleaned):
        cleaned = f"{cleaned} {year}"

    date_formats = [
        "%d %B %Y",       # "8 March 2024"
        "%d %b %Y",       # "8 Mar 2024"
        "%d/%m/%Y",       # "08/03/2024"
        "%Y-%m-%d",       # "2024-03-08"
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Match block parser
# ---------------------------------------------------------------------------

def _empty_match() -> dict[str, Any]:
    """Return a match dict initialised with ``None`` / empty defaults."""
    return {
        "home_team": None,
        "away_team": None,
        "home_score": None,
        "away_score": None,
        "result_text": None,
        "home_try_scorers": [],
        "away_try_scorers": [],
        "home_goal_kickers": [],
        "away_goal_kickers": [],
        "home_field_goals": [],
        "away_field_goals": [],
        "venue": None,
        "date": None,
        "parsed_date": None,
        "kickoff_time": None,
        "halftime_home": None,
        "halftime_away": None,
        "penalty_home": None,
        "penalty_away": None,
        "referee": None,
        "attendance": None,
        "home_lineup": [],
        "home_bench": [],
        "away_lineup": [],
        "away_bench": [],
        "is_bye": False,
        "is_walkover": False,
        "is_abandoned": False,
        "match_url": None,
    }


def parse_match_block(
    block: Tag | str,
    *,
    year: Optional[int] = None,
) -> dict[str, Any]:
    """Parse a single match block (``<div>`` or raw HTML string) into a dict.

    Parameters
    ----------
    block:
        A BeautifulSoup ``Tag`` containing the match div, or a raw HTML
        string for that block.
    year:
        Season year (used to complete date parsing since RLP omits the year
        from date lines).

    Returns
    -------
    dict
        Match data with keys defined in :data:`MATCH_FIELDS`.
    """
    match = _empty_match()

    # Normalise to a Tag.
    if isinstance(block, str):
        soup = BeautifulSoup(block, "lxml")
    else:
        soup = block

    text = soup.get_text(separator=" ")

    # ---- Match URL (the rlplnk ">" link) ---------------------------------
    rlp_link = soup.find("a", class_="rlplnk")
    if rlp_link:
        href = rlp_link.get("href", "")
        if href:
            match["match_url"] = str(href)

    # ---- Edge cases: bye / walkover / abandoned ---------------------------
    text_lower = text.lower()
    if "bye" in text_lower and "had the bye" in text_lower:
        match["is_bye"] = True
        bye_match = re.search(r"(?P<team>.+?)\s+had\s+the\s+bye", text, re.IGNORECASE)
        if bye_match:
            match["home_team"] = bye_match.group("team").strip()
        return match

    if "walkover" in text_lower or "forfeit" in text_lower:
        match["is_walkover"] = True

    if "abandoned" in text_lower or "cancelled" in text_lower:
        match["is_abandoned"] = True

    # ---- Team names and scores from <strong> tags -------------------------
    # The match block has two <strong> tags (at the top level, not inside
    # lineup divs).  The first is home, the second is away.
    # Format: <strong><a href="...">TeamName</a> Score</strong>
    _parse_teams_and_scores(soup, match)

    # ---- Result verb (defeated, drew with, etc.) --------------------------
    verb_m = _RESULT_VERB_RE.search(text)
    if verb_m:
        match["result_text"] = verb_m.group(1).strip().lower()

    # ---- Scorers (from the inline parentheticals) -------------------------
    _parse_inline_scorers(soup, match)

    # ---- Venue (from the "at <a>Venue</a>" pattern) -----------------------
    _parse_venue(soup, match)

    # ---- Metadata line (Date, Kickoff, Halftime, Penalties, Referee, Crowd)
    _parse_metadata_line(text, match, year)

    # ---- Lineups from <div class="quiet small"> blocks --------------------
    _parse_lineup_divs(soup, match)

    return match


def _parse_teams_and_scores(soup: Tag, match: dict[str, Any]) -> None:
    """Extract team names and scores from the <strong> tags.

    The match block contains <strong><a>TeamName</a> Score</strong> pairs.
    We need only the top-level <strong> tags, not those inside lineup divs.
    """
    # Find all <strong> tags that contain a team link.
    strong_tags: list[Tag] = []
    for s in soup.find_all("strong"):
        # Skip <strong> tags inside "quiet small" lineup divs.
        parent = s.parent
        if parent and isinstance(parent, Tag):
            parent_classes = parent.get("class", [])
            if "small" in parent_classes:
                continue
        # Must contain a link to be a team+score tag.
        link = s.find("a")
        if link:
            strong_tags.append(s)

    if len(strong_tags) >= 1:
        home_strong = strong_tags[0]
        home_link = home_strong.find("a")
        if home_link:
            match["home_team"] = home_link.get_text(strip=True)
        # Score is the text after the link inside <strong>.
        home_text = home_strong.get_text(strip=True)
        home_team_text = home_link.get_text(strip=True) if home_link else ""
        score_text = home_text.replace(home_team_text, "", 1).strip()
        match["home_score"] = _safe_int(score_text)

    if len(strong_tags) >= 2:
        away_strong = strong_tags[1]
        away_link = away_strong.find("a")
        if away_link:
            match["away_team"] = away_link.get_text(strip=True)
        away_text = away_strong.get_text(strip=True)
        away_team_text = away_link.get_text(strip=True) if away_link else ""
        score_text = away_text.replace(away_team_text, "", 1).strip()
        match["away_score"] = _safe_int(score_text)


def _parse_inline_scorers(soup: Tag, match: dict[str, Any]) -> None:
    """Extract scorers from the inline parenthetical text.

    The HTML pattern is::

        <strong>Home Score</strong>
        (scorer links tries; kicker N goals) defeated
        <strong>Away Score</strong>
        (scorer links tries; kicker N goals)

    We get the text between each <strong> tag and the next element to find
    the parentheticals.
    """
    # Get the raw HTML of the block (excluding lineup divs).
    # We'll work with the top portion of the block (before the lineup divs).
    block_html = str(soup)

    # Extract the result line portion (before the first <div class="quiet small">)
    result_section = re.split(
        r'<div\s+class="quiet\s+small"', block_html, maxsplit=1, flags=re.IGNORECASE
    )[0]

    # Also strip out the <br/> line (metadata) for cleaner parsing.
    result_text = BeautifulSoup(result_section, "lxml").get_text()

    # Find parenthetical sections.
    parens = re.findall(r"\(([^)]+)\)", result_text)

    if len(parens) >= 1:
        # First parenthetical = home team scorers.
        home_tries, home_goals, home_fgs = _parse_scorers_from_text(parens[0])
        match["home_try_scorers"] = home_tries
        match["home_goal_kickers"] = home_goals
        match["home_field_goals"] = home_fgs

    if len(parens) >= 2:
        # Second parenthetical = away team scorers.
        away_tries, away_goals, away_fgs = _parse_scorers_from_text(parens[1])
        match["away_try_scorers"] = away_tries
        match["away_goal_kickers"] = away_goals
        match["away_field_goals"] = away_fgs


def _parse_venue(soup: Tag, match: dict[str, Any]) -> None:
    """Extract venue from the 'at <a>Venue</a>' pattern.

    The venue link typically appears just before the ``<br/>`` tag in the
    result line.
    """
    # Get the HTML before the lineup divs.
    block_html = str(soup)
    result_section = re.split(
        r'<div\s+class="quiet\s+small"', block_html, maxsplit=1, flags=re.IGNORECASE
    )[0]

    # Look for "at <a ...>VenueName</a>" pattern in the result section.
    venue_match = re.search(
        r'\bat\s+<a[^>]*>([^<]+)</a>',
        result_section,
        re.IGNORECASE,
    )
    if venue_match:
        match["venue"] = venue_match.group(1).strip()


def _parse_metadata_line(
    text: str, match: dict[str, Any], year: Optional[int]
) -> None:
    """Parse the metadata line for date, kickoff, halftime, penalties, etc.

    The metadata line looks like (with &nbsp; replaced by spaces)::

        Date: Sat, 2nd March.  Kickoff: 6:30 PM.  Halftime: Manly 12-10.
        Penalties: Manly 5-1.  Referee: Ashley Klein.  Crowd: 40,746.
    """
    # ---- Date ----------------------------------------------------------
    date_m = _DATE_RE.search(text)
    if date_m:
        match["date"] = date_m.group(0).strip()
        match["parsed_date"] = _parse_date(date_m.group("date"), year)

    # ---- Kickoff -------------------------------------------------------
    kickoff_m = _KICKOFF_RE.search(text)
    if kickoff_m:
        match["kickoff_time"] = kickoff_m.group("time").strip()

    # ---- Halftime ------------------------------------------------------
    ht_m = _HALFTIME_RE.search(text)
    if ht_m:
        match["halftime_home"] = _safe_int(ht_m.group("home"))
        match["halftime_away"] = _safe_int(ht_m.group("away"))

    # ---- Penalties -----------------------------------------------------
    pen_m = _PENALTIES_RE.search(text)
    if pen_m:
        home_val = pen_m.group("home")
        away_val = pen_m.group("away")
        match["penalty_home"] = _safe_int(home_val)
        if away_val and away_val.lower() == "all":
            # "N-all" means both teams had the same count.
            match["penalty_away"] = match["penalty_home"]
        else:
            match["penalty_away"] = _safe_int(away_val)

    # ---- Referee -------------------------------------------------------
    # The referee name is inside a link: <a class="quiet" href="...">Name</a>
    # after "Referee:".  We extract from text as the link text is rendered.
    ref_m = re.search(
        r"Referee:\s*(?P<ref>[A-Za-z][\w\s\-'.]+?)(?:\.|,|\s{2,}|$)",
        text,
        re.IGNORECASE,
    )
    if ref_m:
        match["referee"] = ref_m.group("ref").strip()

    # ---- Attendance ----------------------------------------------------
    crowd_m = _CROWD_RE.search(text)
    if crowd_m:
        match["attendance"] = _safe_int(crowd_m.group("crowd"))


def _parse_lineup_divs(soup: Tag, match: dict[str, Any]) -> None:
    """Extract lineups from <div class="quiet small"> blocks.

    Each team's lineup is in a separate div with the structure::

        <div class="quiet small">
        <strong><a href="...">TeamName</a>:</strong>
        <a>Player1</a>, <a>Player2</a>, ...; ... <em>Int:</em>
        <a>Bench1</a>, <a>Bench2</a>, ...
        </div>

    Semicolons separate positional groups (backs; halves; forwards).
    """
    home_team = match.get("home_team") or ""
    away_team = match.get("away_team") or ""

    lineup_divs = soup.find_all("div", class_="small")
    lineup_count = 0

    for div in lineup_divs:
        # Identify team from the <strong> tag inside.
        strong = div.find("strong")
        if not strong:
            continue

        team_link = strong.find("a")
        div_team_name = ""
        if team_link:
            div_team_name = team_link.get_text(strip=True).rstrip(":")
        else:
            div_team_name = strong.get_text(strip=True).rstrip(":")

        # Determine if this is home or away.
        is_home = (
            div_team_name.lower() == home_team.lower()
            if home_team else False
        )
        is_away = (
            div_team_name.lower() == away_team.lower()
            if away_team else False
        )

        # If neither matches exactly, use order (first = home, second = away).
        if not is_home and not is_away:
            if lineup_count == 0:
                is_home = True
            else:
                is_away = True
        lineup_count += 1

        # Extract player names from <a> tags (after the <strong> tag).
        # Split on <em>Int:</em> to separate starters from bench.
        int_tag = div.find("em")
        starters: list[str] = []
        bench: list[str] = []

        if int_tag:
            # All <a> tags before <em>Int:</em> are starters,
            # all after are bench.
            in_bench = False
            for child in div.descendants:
                if child is int_tag or (isinstance(child, Tag) and child is int_tag):
                    in_bench = True
                    continue
                if isinstance(child, Tag) and child.name == "a" and child.parent is not strong:
                    # Skip the team link inside <strong>.
                    player_name = child.get_text(strip=True)
                    # Filter out non-player links (team links inside strong).
                    if player_name and child.find_parent("strong") is None:
                        if player_name.endswith(":"):
                            continue  # Skip team name links
                        # Strip "(C)" captain marker.
                        player_name = re.sub(r"\s*\(C\)\s*$", "", player_name).strip()
                        if in_bench:
                            bench.append(player_name)
                        else:
                            starters.append(player_name)
        else:
            # No Int tag; all players are starters.
            for a_tag in div.find_all("a"):
                if a_tag.find_parent("strong") is not None:
                    continue
                player_name = a_tag.get_text(strip=True)
                if player_name and not player_name.endswith(":"):
                    player_name = re.sub(r"\s*\(C\)\s*$", "", player_name).strip()
                    starters.append(player_name)

        if is_home:
            match["home_lineup"] = starters
            match["home_bench"] = bench
        elif is_away:
            match["away_lineup"] = starters
            match["away_bench"] = bench


# ---------------------------------------------------------------------------
# Full round-summary page parser
# ---------------------------------------------------------------------------

def parse_round_summary(
    html: str,
    year: Optional[int] = None,
    round_id: Optional[int | str] = None,
) -> list[dict[str, Any]]:
    """Parse all matches from an RLP round-summary HTML page.

    Parameters
    ----------
    html:
        Raw HTML content of the round-summary page.
    year:
        Season year (attached to each match dict for convenience).
    round_id:
        Round identifier (attached to each match dict).

    Returns
    -------
    list[dict]
        One dict per match found on the page.  Byes are included with
        ``is_bye=True``.  If the page contains no recognisable match
        blocks, an empty list is returned.
    """
    soup = BeautifulSoup(html, "lxml")
    matches: list[dict[str, Any]] = []

    # Strategy 1 (primary): Find match blocks by their characteristic style.
    # RLP match blocks are <div> elements with inline style containing
    # "font-size:130%" and "border-bottom:dotted".
    blocks: list[Tag] = []
    for div in soup.find_all("div"):
        style = div.get("style", "")
        if "font-size:130%" in style or "font-size: 130%" in style:
            if "border-bottom" in style:
                blocks.append(div)

    # Strategy 2: Also look for divs with class "quiet" that have the
    # right structure (contain <strong> tags with team links and have
    # "defeated"/"drew with" in the text).
    if not blocks:
        for div in soup.find_all("div", class_="quiet"):
            div_classes = div.get("class", [])
            # Skip "quiet small" lineup divs.
            if "small" in div_classes:
                continue
            strong_tags = div.find_all("strong", recursive=False)
            if strong_tags:
                text = div.get_text()
                if _RESULT_VERB_RE.search(text) or "had the bye" in text.lower():
                    blocks.append(div)

    if not blocks:
        logger.warning(
            "No match blocks found on round-summary page "
            "(year=%s, round=%s).",
            year,
            round_id,
        )
        return matches

    # Parse each block.
    for block in blocks:
        m = parse_match_block(block, year=year)
        _annotate(m, year, round_id)

        # Skip truly empty blocks (some pages have decorative divs).
        if m["home_team"] is None and not m["is_bye"]:
            continue

        matches.append(m)

    logger.info(
        "Parsed %d matches from round-summary page (year=%s, round=%s).",
        len(matches),
        year,
        round_id,
    )
    return matches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annotate(
    match: dict[str, Any],
    year: Optional[int],
    round_id: Optional[int | str],
) -> None:
    """Attach season/round metadata to a parsed match dict."""
    if year is not None:
        match["year"] = year
    if round_id is not None:
        match["round"] = round_id
