"""
Polite scraping utilities: rate limiting, caching, and retry logic.

This module ensures the RLP scraper behaves respectfully:
- Enforces a configurable minimum delay between HTTP requests.
- Caches every fetched HTML page locally so that repeated runs never
  re-download the same page.
- Retries transient failures with exponential back-off.
- Sends a descriptive ``User-Agent`` header identifying the project.

Cache layout
------------
HTML files are stored under two parallel directories:

- ``scraping/cache/`` -- fast lookup cache keyed by the URL path.
- ``data/raw/rlp/``   -- long-term archival copy using the same key.

Both directories mirror the URL path structure, e.g. the page at
``/seasons/nrl-2024/round-1/summary.html`` is cached as
``scraping/cache/seasons/nrl-2024/round-1/summary.html``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from config.settings import (
    PROJECT_ROOT,
    RAW_DIR,
    RLP_BASE_URL,
    SCRAPE_DELAY_SECONDS,
    USER_AGENT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default cache directories
# ---------------------------------------------------------------------------
SCRAPING_CACHE_DIR: Path = PROJECT_ROOT / "scraping" / "cache"
RAW_RLP_DIR: Path = RAW_DIR / "rlp"

# Ensure they exist.
SCRAPING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_RLP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cache-key derivation
# ---------------------------------------------------------------------------

def _url_to_cache_path(url: str, cache_root: Path) -> Path:
    """Map a full URL to a local file path under *cache_root*.

    The URL path (everything after the host) is used directly as the
    relative file path so the cache mirrors the site structure.  Query
    strings, if any, are incorporated via a short hash suffix to avoid
    collisions.

    Parameters
    ----------
    url:
        Fully-qualified URL (e.g.
        ``https://www.rugbyleagueproject.org/seasons/nrl-2024/round-1/summary.html``).
    cache_root:
        Directory under which the cached file will be stored.

    Returns
    -------
    Path
        Absolute path to the cached HTML file.
    """
    parsed = urlparse(url)
    # Strip leading slash so Path joining works correctly.
    rel = parsed.path.lstrip("/")

    if parsed.query:
        # Disambiguate URLs that differ only by query string.
        query_hash = hashlib.md5(parsed.query.encode()).hexdigest()[:8]
        stem = Path(rel).stem
        suffix = Path(rel).suffix or ".html"
        rel = str(Path(rel).parent / f"{stem}_{query_hash}{suffix}")

    return cache_root / rel


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Enforces a minimum delay between successive HTTP requests.

    Parameters
    ----------
    delay:
        Minimum number of seconds between requests.  Defaults to the
        value of :data:`config.settings.SCRAPE_DELAY_SECONDS` (2.5 s).
    """

    def __init__(self, delay: float = SCRAPE_DELAY_SECONDS) -> None:
        self.delay = delay
        self._last_request_time: float = 0.0

    def wait(self) -> None:
        """Block until enough time has elapsed since the last request."""
        elapsed = time.monotonic() - self._last_request_time
        remaining = self.delay - elapsed
        if remaining > 0:
            logger.debug("Rate limiter: sleeping %.2f s", remaining)
            time.sleep(remaining)
        self._last_request_time = time.monotonic()

    def stamp(self) -> None:
        """Record that a request was just made (without sleeping)."""
        self._last_request_time = time.monotonic()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _read_cache(url: str) -> Optional[str]:
    """Return cached HTML for *url*, or ``None`` if not cached.

    Checks the scraping cache first, then the raw-data archive.
    """
    for root in (SCRAPING_CACHE_DIR, RAW_RLP_DIR):
        path = _url_to_cache_path(url, root)
        if path.is_file():
            logger.debug("Cache hit (%s): %s", root.name, url)
            return path.read_text(encoding="utf-8")
    return None


def _write_cache(url: str, html: str) -> None:
    """Persist *html* to both cache directories."""
    for root in (SCRAPING_CACHE_DIR, RAW_RLP_DIR):
        path = _url_to_cache_path(url, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    logger.debug("Cached: %s", url)


def is_cached(url: str) -> bool:
    """Return ``True`` if *url* has a local cache entry."""
    return _read_cache(url) is not None


def cache_path_for(url: str) -> Path:
    """Return the primary (scraping/cache) file path for *url*.

    Useful for debugging or manual inspection of cached files.
    """
    return _url_to_cache_path(url, SCRAPING_CACHE_DIR)


def clear_cache(url: str) -> None:
    """Remove cached copies of *url* from both cache locations."""
    for root in (SCRAPING_CACHE_DIR, RAW_RLP_DIR):
        path = _url_to_cache_path(url, root)
        if path.is_file():
            path.unlink()
            logger.info("Deleted cache file: %s", path)


# ---------------------------------------------------------------------------
# Fetch with caching + rate limiting + retries
# ---------------------------------------------------------------------------

_DEFAULT_LIMITER = RateLimiter()

_SESSION: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Return a reusable :class:`requests.Session` with the project UA."""
    global _SESSION  # noqa: PLW0603
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({"User-Agent": USER_AGENT})
    return _SESSION


def fetch_url(
    url: str,
    *,
    use_cache: bool = True,
    rate_limiter: RateLimiter | None = None,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    timeout: float = 30.0,
) -> str:
    """Fetch *url* and return its HTML content as a string.

    Behaviour
    ---------
    1. If *use_cache* is ``True`` and a cached copy exists, return it
       immediately (no network I/O, no rate-limit wait).
    2. Otherwise, wait for the rate limiter, then issue an HTTP GET.
    3. On transient failures (5xx, connection errors, timeouts), retry up
       to *max_retries* times with exponential back-off.
    4. On success, cache the response body and return it.

    Parameters
    ----------
    url:
        Fully-qualified URL to fetch.
    use_cache:
        Set to ``False`` to force a fresh download.
    rate_limiter:
        A :class:`RateLimiter` instance.  If ``None``, uses the module-
        level default (2.5 s delay).
    max_retries:
        Maximum number of retry attempts for transient errors.
    backoff_base:
        Base for exponential back-off between retries (seconds).
        Wait time = ``backoff_base ** attempt``.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    str
        The HTML content of the page.

    Raises
    ------
    requests.HTTPError
        If the server returns a non-retryable 4xx status code.
    requests.ConnectionError
        If all retry attempts are exhausted.
    """
    # 1. Cache lookup
    if use_cache:
        cached = _read_cache(url)
        if cached is not None:
            return cached

    limiter = rate_limiter or _DEFAULT_LIMITER
    session = _get_session()

    last_exception: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            limiter.wait()
            logger.info("GET %s (attempt %d/%d)", url, attempt + 1, max_retries + 1)
            response = session.get(url, timeout=timeout)

            # Non-retryable client errors (except 429).
            if 400 <= response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()

            # Retryable server / rate-limit errors.
            if response.status_code >= 500 or response.status_code == 429:
                wait = backoff_base ** attempt
                logger.warning(
                    "HTTP %d for %s -- retrying in %.1f s",
                    response.status_code,
                    url,
                    wait,
                )
                time.sleep(wait)
                continue

            # Success path.
            response.raise_for_status()
            html = response.text

            # 2. Cache the result.
            _write_cache(url, html)
            return html

        except requests.ConnectionError as exc:
            last_exception = exc
            wait = backoff_base ** attempt
            logger.warning(
                "Connection error for %s -- retrying in %.1f s: %s",
                url,
                wait,
                exc,
            )
            time.sleep(wait)

        except requests.Timeout as exc:
            last_exception = exc
            wait = backoff_base ** attempt
            logger.warning(
                "Timeout for %s -- retrying in %.1f s", url, wait
            )
            time.sleep(wait)

    # All retries exhausted.
    raise requests.ConnectionError(
        f"Failed to fetch {url} after {max_retries + 1} attempts. "
        f"Last error: {last_exception}"
    )


# ---------------------------------------------------------------------------
# Convenience: batch fetch with progress reporting
# ---------------------------------------------------------------------------

def fetch_urls(
    urls: list[str],
    *,
    use_cache: bool = True,
    rate_limiter: RateLimiter | None = None,
    max_retries: int = 3,
) -> dict[str, str]:
    """Fetch multiple URLs and return a ``{url: html}`` mapping.

    Useful for simple batch jobs where the caller does not need per-URL
    progress bars (the scraper module uses tqdm for that).

    Parameters
    ----------
    urls:
        List of URLs to fetch.
    use_cache:
        Whether to honour the local cache.
    rate_limiter:
        Optional rate limiter override.
    max_retries:
        Maximum retries per URL.

    Returns
    -------
    dict[str, str]
        Mapping of URL to its HTML content.  URLs that failed after
        all retries are omitted with a warning.
    """
    results: dict[str, str] = {}
    for url in urls:
        try:
            results[url] = fetch_url(
                url,
                use_cache=use_cache,
                rate_limiter=rate_limiter,
                max_retries=max_retries,
            )
        except (requests.HTTPError, requests.ConnectionError) as exc:
            logger.error("Skipping %s: %s", url, exc)
    return results
