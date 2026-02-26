"""
Scraping package for the NRL Match Winner Prediction project.

Submodules
----------
rlp_url_builder     URL generation helpers for Rugby League Project pages.
rate_limiter        Polite scraping: rate limiting, caching, retry logic.
rlp_scraper         Core HTML scraper orchestrator.
rlp_match_parser    Parse match details from RLP round-summary pages.
rlp_ladder_parser   Parse ladder/standings tables from RLP round-ladder pages.
odds_loader         Load and clean AusSportsBetting historical odds Excel.
odds_api            Fetch live fixtures and odds from The Odds API.
nrl_teamlists       Fetch team lists from the NRL.com official API.
"""
