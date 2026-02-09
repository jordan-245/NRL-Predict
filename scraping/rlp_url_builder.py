"""
URL generation helpers for Rugby League Project (rugbyleagueproject.org).

Builds fully-qualified URLs for every RLP page type used by the scraping
pipeline.  All functions return plain strings; no network I/O occurs here.

URL patterns (base: https://www.rugbyleagueproject.org):
    Season summary   /seasons/nrl-{year}/summary.html
    Season results   /seasons/nrl-{year}/results.html
    Season players   /seasons/nrl-{year}/players.html
    Round summary    /seasons/nrl-{year}/round-{N}/summary.html
    Round ladder     /seasons/nrl-{year}/round-{N}/ladder.html
    Match summary    /seasons/nrl-{year}/round-{N}/{home}-vs-{away}/summary.html
    Match stats      /seasons/nrl-{year}/round-{N}/{home}-vs-{away}/stats.html
    Player profile   /players/{player-slug}/summary.html
"""

from __future__ import annotations

from config.settings import FINALS_ROUNDS, RLP_BASE_URL

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round_slug(round_id: int | str) -> str:
    """Convert a round identifier to its URL path segment.

    Regular-season rounds (integers) become ``"round-{N}"``.
    Finals-round strings (e.g. ``"qualif-final"``) are returned as-is.

    Parameters
    ----------
    round_id:
        An integer round number (1-27) or a finals-round slug such as
        ``"qualif-final"``, ``"elim-final"``, ``"semi-final"``,
        ``"prelim-final"``, or ``"grand-final"``.

    Returns
    -------
    str
        The URL path segment for the round.

    Raises
    ------
    ValueError
        If *round_id* is a string not present in :data:`FINALS_ROUNDS`.
    """
    if isinstance(round_id, int):
        return f"round-{round_id}"
    if isinstance(round_id, str):
        if round_id in FINALS_ROUNDS:
            return round_id
        raise ValueError(
            f"Unknown finals round slug {round_id!r}. "
            f"Expected one of {FINALS_ROUNDS}."
        )
    raise TypeError(
        f"round_id must be int or str, got {type(round_id).__name__}"
    )


def _match_slug(home_slug: str, away_slug: str) -> str:
    """Build the ``{home}-vs-{away}`` path segment.

    Parameters
    ----------
    home_slug:
        Lowercase-hyphenated full team name (e.g. ``"melbourne-storm"``).
    away_slug:
        Lowercase-hyphenated full team name (e.g. ``"penrith-panthers"``).

    Returns
    -------
    str
        Combined slug in the form ``"melbourne-storm-vs-penrith-panthers"``.
    """
    return f"{home_slug}-vs-{away_slug}"


# ---------------------------------------------------------------------------
# Team-name slug conversion
# ---------------------------------------------------------------------------

def team_name_to_slug(team_name: str) -> str:
    """Convert a human-readable team name to the RLP URL slug format.

    The slug is *lowercase*, with spaces and special punctuation replaced by
    hyphens.

    Examples
    --------
    >>> team_name_to_slug("Melbourne Storm")
    'melbourne-storm'
    >>> team_name_to_slug("South Sydney Rabbitohs")
    'south-sydney-rabbitohs'
    >>> team_name_to_slug("Canterbury-Bankstown Bulldogs")
    'canterbury-bankstown-bulldogs'
    >>> team_name_to_slug("St George Illawarra Dragons")
    'st-george-illawarra-dragons'
    """
    slug = team_name.strip().lower()
    # Replace common punctuation that appears in NRL team names.
    slug = slug.replace(".", "")
    # Normalise whitespace and existing hyphens to a single hyphen.
    slug = slug.replace(" ", "-")
    # Collapse multiple consecutive hyphens.
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug


# ---------------------------------------------------------------------------
# Public URL builders
# ---------------------------------------------------------------------------

def season_summary_url(year: int, *, base_url: str = RLP_BASE_URL) -> str:
    """Return the URL for a season's summary page.

    Parameters
    ----------
    year:
        The NRL season year (e.g. 2024).
    base_url:
        Override the default RLP base URL for testing.
    """
    return f"{base_url}/seasons/nrl-{year}/summary.html"


def season_results_url(year: int, *, base_url: str = RLP_BASE_URL) -> str:
    """Return the URL for a season's full results list."""
    return f"{base_url}/seasons/nrl-{year}/results.html"


def season_players_url(year: int, *, base_url: str = RLP_BASE_URL) -> str:
    """Return the URL for a season's player listing."""
    return f"{base_url}/seasons/nrl-{year}/players.html"


def round_summary_url(
    year: int,
    round_id: int | str,
    *,
    base_url: str = RLP_BASE_URL,
) -> str:
    """Return the URL for a round's summary page.

    Parameters
    ----------
    year:
        NRL season year.
    round_id:
        Round number (1-27) or finals-round slug.
    base_url:
        Override the default RLP base URL for testing.
    """
    slug = _round_slug(round_id)
    return f"{base_url}/seasons/nrl-{year}/{slug}/summary.html"


def round_ladder_url(
    year: int,
    round_id: int | str,
    *,
    base_url: str = RLP_BASE_URL,
) -> str:
    """Return the URL for the ladder after a given round."""
    slug = _round_slug(round_id)
    return f"{base_url}/seasons/nrl-{year}/{slug}/ladder.html"


def match_summary_url(
    year: int,
    round_id: int | str,
    home_slug: str,
    away_slug: str,
    *,
    base_url: str = RLP_BASE_URL,
) -> str:
    """Return the URL for an individual match's summary page.

    Parameters
    ----------
    year:
        NRL season year.
    round_id:
        Round number or finals-round slug.
    home_slug:
        Home team slug (e.g. ``"melbourne-storm"``).
    away_slug:
        Away team slug (e.g. ``"penrith-panthers"``).
    base_url:
        Override the default RLP base URL for testing.
    """
    rslug = _round_slug(round_id)
    mslug = _match_slug(home_slug, away_slug)
    return f"{base_url}/seasons/nrl-{year}/{rslug}/{mslug}/summary.html"


def match_stats_url(
    year: int,
    round_id: int | str,
    home_slug: str,
    away_slug: str,
    *,
    base_url: str = RLP_BASE_URL,
) -> str:
    """Return the URL for an individual match's detailed stats page."""
    rslug = _round_slug(round_id)
    mslug = _match_slug(home_slug, away_slug)
    return f"{base_url}/seasons/nrl-{year}/{rslug}/{mslug}/stats.html"


def player_profile_url(
    player_slug: str,
    *,
    base_url: str = RLP_BASE_URL,
) -> str:
    """Return the URL for a player's profile/summary page.

    Parameters
    ----------
    player_slug:
        The player's hyphenated slug as used by RLP
        (e.g. ``"nathan-cleary"``).
    base_url:
        Override the default RLP base URL for testing.
    """
    return f"{base_url}/players/{player_slug}/summary.html"


# ---------------------------------------------------------------------------
# Convenience: bulk URL generation
# ---------------------------------------------------------------------------

def all_round_summary_urls(
    year: int,
    regular_rounds: range | None = None,
    finals_rounds: list[str] | None = None,
    *,
    base_url: str = RLP_BASE_URL,
) -> list[tuple[int | str, str]]:
    """Return ``(round_id, url)`` pairs for every round in a season.

    Parameters
    ----------
    year:
        NRL season year.
    regular_rounds:
        Override the default ``range(1, 28)`` for seasons with a different
        number of regular rounds.
    finals_rounds:
        Override the default list of finals-round slugs.
    base_url:
        Override the default RLP base URL for testing.

    Returns
    -------
    list[tuple[int | str, str]]
        Ordered list of ``(round_id, url)`` for all rounds.
    """
    from config.settings import FINALS_ROUNDS as _DEFAULT_FINALS
    from config.settings import REGULAR_ROUNDS as _DEFAULT_REGULAR

    _regular = regular_rounds if regular_rounds is not None else _DEFAULT_REGULAR
    _finals = finals_rounds if finals_rounds is not None else _DEFAULT_FINALS

    urls: list[tuple[int | str, str]] = []
    for r in _regular:
        urls.append((r, round_summary_url(year, r, base_url=base_url)))
    for f in _finals:
        urls.append((f, round_summary_url(year, f, base_url=base_url)))
    return urls


def all_round_ladder_urls(
    year: int,
    regular_rounds: range | None = None,
    *,
    base_url: str = RLP_BASE_URL,
) -> list[tuple[int, str]]:
    """Return ``(round_number, url)`` pairs for every regular-season round ladder.

    Finals rounds typically don't have separate ladder pages, so only
    regular-season rounds are included by default.
    """
    from config.settings import REGULAR_ROUNDS as _DEFAULT_REGULAR

    _regular = regular_rounds if regular_rounds is not None else _DEFAULT_REGULAR

    return [
        (r, round_ladder_url(year, r, base_url=base_url))
        for r in _regular
    ]
