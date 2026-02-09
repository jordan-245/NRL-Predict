"""
Core RLP HTML scraper.

Orchestrates the downloading of round-summary, round-ladder, season-player,
and match-stats pages from Rugby League Project.  Returns **raw HTML strings**;
all parsing is delegated to the dedicated parser modules
(:mod:`rlp_match_parser`, :mod:`rlp_ladder_parser`, :mod:`rlp_player_parser`).

Usage
-----
::

    from scraping.rlp_scraper import RLPScraper

    scraper = RLPScraper()

    # Single round
    html = scraper.scrape_round_summary(2024, 1)

    # Full season (regular + finals) with progress bar
    htmls = scraper.scrape_season_round_summaries(2024)
"""

from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from config.settings import (
    ALL_ROUNDS,
    END_YEAR,
    FINALS_ROUNDS,
    REGULAR_ROUNDS,
    START_YEAR,
)
from scraping.rate_limiter import RateLimiter, fetch_url
from scraping.rlp_url_builder import (
    match_stats_url,
    player_profile_url,
    round_ladder_url,
    round_summary_url,
    season_players_url,
)

logger = logging.getLogger(__name__)


class RLPScraper:
    """High-level interface for scraping Rugby League Project pages.

    Parameters
    ----------
    rate_limiter:
        Custom :class:`~scraping.rate_limiter.RateLimiter`.  If ``None``,
        the module default (2.5 s delay) is used.
    use_cache:
        If ``True`` (the default), return locally cached pages instead
        of re-downloading them.
    max_retries:
        Number of retry attempts per HTTP request.
    show_progress:
        If ``True`` (the default), display tqdm progress bars during
        bulk scrape operations.
    """

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        *,
        use_cache: bool = True,
        max_retries: int = 3,
        show_progress: bool = True,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.show_progress = show_progress

    # ------------------------------------------------------------------
    # Internal fetch wrapper
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> Optional[str]:
        """Fetch a single URL; return ``None`` on 404 (expected for
        rounds/matches that don't exist)."""
        try:
            return fetch_url(
                url,
                use_cache=self.use_cache,
                rate_limiter=self.rate_limiter,
                max_retries=self.max_retries,
            )
        except Exception as exc:
            # 404s are expected for rounds that don't exist in a season
            # (e.g. finals week pages for teams that were eliminated).
            if "404" in str(exc):
                logger.debug("404 (expected): %s", url)
                return None
            logger.error("Failed to fetch %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Single-page methods
    # ------------------------------------------------------------------

    def scrape_round_summary(
        self, year: int, round_id: int | str
    ) -> Optional[str]:
        """Fetch the round-summary page for one round.

        Parameters
        ----------
        year:
            NRL season year.
        round_id:
            Round number (1-27) or finals slug (e.g. ``"grand-final"``).

        Returns
        -------
        str or None
            Raw HTML string, or ``None`` if the page does not exist.
        """
        url = round_summary_url(year, round_id)
        return self._fetch(url)

    def scrape_round_ladder(
        self, year: int, round_id: int | str
    ) -> Optional[str]:
        """Fetch the ladder page for one round.

        Parameters
        ----------
        year:
            NRL season year.
        round_id:
            Round number or finals slug.

        Returns
        -------
        str or None
            Raw HTML string, or ``None`` if unavailable.
        """
        url = round_ladder_url(year, round_id)
        return self._fetch(url)

    def scrape_season_players(self, year: int) -> Optional[str]:
        """Fetch the season-level player listing.

        Parameters
        ----------
        year:
            NRL season year.

        Returns
        -------
        str or None
            Raw HTML string, or ``None`` if unavailable.
        """
        url = season_players_url(year)
        return self._fetch(url)

    def scrape_match_stats(
        self,
        year: int,
        round_id: int | str,
        home_slug: str,
        away_slug: str,
    ) -> Optional[str]:
        """Fetch the detailed stats page for a specific match.

        Parameters
        ----------
        year:
            NRL season year.
        round_id:
            Round number or finals slug.
        home_slug:
            Home team URL slug (e.g. ``"melbourne-storm"``).
        away_slug:
            Away team URL slug (e.g. ``"penrith-panthers"``).

        Returns
        -------
        str or None
            Raw HTML string, or ``None`` if unavailable.
        """
        url = match_stats_url(year, round_id, home_slug, away_slug)
        return self._fetch(url)

    def scrape_player_profile(self, player_slug: str) -> Optional[str]:
        """Fetch a player's profile page.

        Parameters
        ----------
        player_slug:
            Player URL slug (e.g. ``"nathan-cleary"``).

        Returns
        -------
        str or None
            Raw HTML string, or ``None`` if unavailable.
        """
        url = player_profile_url(player_slug)
        return self._fetch(url)

    # ------------------------------------------------------------------
    # Bulk / season-level methods
    # ------------------------------------------------------------------

    def scrape_season_round_summaries(
        self,
        year: int,
        rounds: list[int | str] | None = None,
    ) -> dict[int | str, str]:
        """Scrape round-summary pages for an entire season.

        Parameters
        ----------
        year:
            NRL season year.
        rounds:
            Specific round identifiers to scrape.  Defaults to
            :data:`config.settings.ALL_ROUNDS` (1-27 + finals).

        Returns
        -------
        dict[int | str, str]
            Mapping of ``{round_id: html}`` for successfully fetched rounds.
        """
        _rounds = rounds if rounds is not None else ALL_ROUNDS
        results: dict[int | str, str] = {}

        desc = f"Round summaries {year}"
        iterator = tqdm(_rounds, desc=desc, disable=not self.show_progress)

        for round_id in iterator:
            iterator.set_postfix(round=str(round_id))
            html = self.scrape_round_summary(year, round_id)
            if html is not None:
                results[round_id] = html

        logger.info(
            "Season %d: fetched %d/%d round summary pages.",
            year,
            len(results),
            len(_rounds),
        )
        return results

    def scrape_season_round_ladders(
        self,
        year: int,
        rounds: list[int] | None = None,
    ) -> dict[int, str]:
        """Scrape round-ladder pages for an entire season.

        Only regular-season rounds are scraped by default since finals
        rounds rarely have standalone ladder pages on RLP.

        Parameters
        ----------
        year:
            NRL season year.
        rounds:
            Specific round numbers to scrape.  Defaults to
            ``range(1, 28)``.

        Returns
        -------
        dict[int, str]
            Mapping of ``{round_number: html}`` for successfully fetched rounds.
        """
        _rounds = rounds if rounds is not None else list(REGULAR_ROUNDS)
        results: dict[int, str] = {}

        desc = f"Ladders {year}"
        iterator = tqdm(_rounds, desc=desc, disable=not self.show_progress)

        for round_num in iterator:
            iterator.set_postfix(round=str(round_num))
            html = self.scrape_round_ladder(year, round_num)
            if html is not None:
                results[round_num] = html

        logger.info(
            "Season %d: fetched %d/%d round ladder pages.",
            year,
            len(results),
            len(_rounds),
        )
        return results

    def scrape_season_match_stats(
        self,
        year: int,
        matches: list[tuple[int | str, str, str]],
    ) -> dict[tuple[int | str, str, str], str]:
        """Scrape match-stats pages for a list of matches.

        Parameters
        ----------
        year:
            NRL season year.
        matches:
            List of ``(round_id, home_slug, away_slug)`` triples.

        Returns
        -------
        dict[tuple[int | str, str, str], str]
            Mapping of ``{(round_id, home, away): html}`` for successful pages.
        """
        results: dict[tuple[int | str, str, str], str] = {}

        desc = f"Match stats {year}"
        iterator = tqdm(matches, desc=desc, disable=not self.show_progress)

        for round_id, home, away in iterator:
            iterator.set_postfix(match=f"{home} v {away}")
            html = self.scrape_match_stats(year, round_id, home, away)
            if html is not None:
                results[(round_id, home, away)] = html

        logger.info(
            "Season %d: fetched %d/%d match stats pages.",
            year,
            len(results),
            len(matches),
        )
        return results

    # ------------------------------------------------------------------
    # Multi-season
    # ------------------------------------------------------------------

    def scrape_all_seasons(
        self,
        start_year: int = START_YEAR,
        end_year: int = END_YEAR,
        *,
        include_ladders: bool = True,
        include_players: bool = True,
    ) -> dict[int, dict[str, object]]:
        """Scrape round summaries (and optionally ladders/players) for
        every season in the configured year range.

        Parameters
        ----------
        start_year:
            First season to scrape (inclusive).
        end_year:
            Last season to scrape (inclusive).
        include_ladders:
            Also scrape round-ladder pages.
        include_players:
            Also scrape season-player pages.

        Returns
        -------
        dict[int, dict]
            Nested dict keyed by year with keys ``"round_summaries"``,
            ``"ladders"`` (if requested), and ``"players"`` (if requested).
        """
        all_data: dict[int, dict[str, object]] = {}

        for year in range(start_year, end_year + 1):
            logger.info("=== Scraping season %d ===", year)
            season: dict[str, object] = {}

            season["round_summaries"] = self.scrape_season_round_summaries(year)

            if include_ladders:
                season["ladders"] = self.scrape_season_round_ladders(year)

            if include_players:
                season["players"] = self.scrape_season_players(year)

            all_data[year] = season

        return all_data
