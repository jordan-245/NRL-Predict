"""
Global configuration for the NRL Match Winner Prediction project.

Loads environment overrides from a .env file in the project root (if present),
then exposes path constants, scraping parameters, modelling defaults, and
round/season definitions used throughout the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # python-dotenv is not installed
    load_dotenv = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Resolve project root as two levels up from this file (config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root; silently continue if it does not exist
# or if python-dotenv is not installed.
_dotenv_path = PROJECT_ROOT / ".env"
if load_dotenv is not None:
    load_dotenv(dotenv_path=_dotenv_path, override=False)

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FEATURES_DIR: Path = DATA_DIR / "features"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# Ensure key directories exist at import time so downstream code can write
# without extra boilerplate.
for _dir in (RAW_DIR, PROCESSED_DIR, FEATURES_DIR, OUTPUTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Scraping configuration
# ---------------------------------------------------------------------------
RLP_BASE_URL: str = os.getenv(
    "RLP_BASE_URL",
    "https://www.rugbyleagueproject.org",
)

SCRAPE_DELAY_SECONDS: float = float(
    os.getenv("SCRAPE_DELAY_SECONDS", "2.5"),
)

USER_AGENT: str = os.getenv(
    "USER_AGENT",
    (
        "NRL-Predictor/0.1 "
        "(+https://github.com/jorda/nrl-predictor; "
        "research project; respects robots.txt)"
    ),
)

# ---------------------------------------------------------------------------
# Season / year boundaries
# ---------------------------------------------------------------------------
START_YEAR: int = int(os.getenv("START_YEAR", "2013"))
END_YEAR: int = int(os.getenv("END_YEAR", "2026"))
PREDICT_YEAR: int = int(os.getenv("PREDICT_YEAR", "2026"))

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------
N_OPTUNA_TRIALS: int = int(os.getenv("N_OPTUNA_TRIALS", "200"))

# ---------------------------------------------------------------------------
# Round definitions
# ---------------------------------------------------------------------------
# Regular-season rounds (1 through 27 inclusive for the standard NRL draw).
REGULAR_ROUNDS: range = range(1, 28)

# Finals-series round slugs, ordered from earliest to latest.
FINALS_ROUNDS: list[str] = [
    "qualif-final",
    "elim-final",
    "semi-final",
    "prelim-final",
    "grand-final",
]

# Combined ordered list of all round identifiers (ints then strings).
ALL_ROUNDS: list[int | str] = list(REGULAR_ROUNDS) + FINALS_ROUNDS
