"""
Scrape referee assignments and ground conditions from NRL.com API.

Creates data/processed/match_officials.parquet with:
  year, round, home_team, away_team, referee, ground_conditions, weather
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NRL_API_BASE = "https://www.nrl.com"
DRAW_API = f"{NRL_API_BASE}/draw/data"
COMPETITION_ID = 111
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIALS_PATH = PROJECT_ROOT / "data" / "processed" / "match_officials.parquet"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "officials"

TEAM_MAP = {
    "Broncos": "Brisbane Broncos", "Raiders": "Canberra Raiders",
    "Bulldogs": "Canterbury Bulldogs", "Sharks": "Cronulla Sharks",
    "Titans": "Gold Coast Titans", "Sea Eagles": "Manly Sea Eagles",
    "Storm": "Melbourne Storm", "Knights": "Newcastle Knights",
    "Cowboys": "North Queensland Cowboys", "Eels": "Parramatta Eels",
    "Panthers": "Penrith Panthers", "Rabbitohs": "South Sydney Rabbitohs",
    "Roosters": "Sydney Roosters", "Dragons": "St George Illawarra Dragons",
    "Warriors": "New Zealand Warriors", "NZ Warriors": "New Zealand Warriors",
    "Wests Tigers": "Wests Tigers", "Tigers": "Wests Tigers",
    "Dolphins": "Dolphins",
}


def _std_team(nick: str) -> str:
    return TEAM_MAP.get(nick, nick)


def fetch_round_officials(year: int, rnd: int, delay: float = 0.3) -> list[dict]:
    """Fetch referee + conditions for all matches in a round."""
    params = {"competition": COMPETITION_ID, "season": year, "round": rnd}
    try:
        r = requests.get(DRAW_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        fixtures = r.json().get("fixtures", [])
    except Exception as e:
        logger.warning("Failed draw %d R%d: %s", year, rnd, e)
        return []

    rows = []
    for f in fixtures:
        state = f.get("matchState", "")
        if state not in ("FullTime", "FullTimeExtraTime"):
            continue
        mc_url = f.get("matchCentreUrl")
        if not mc_url:
            continue

        try:
            r2 = requests.get(f"{NRL_API_BASE}{mc_url}data", headers=HEADERS, timeout=15)
            r2.raise_for_status()
            d = r2.json()
        except Exception:
            continue

        home = _std_team(d.get("homeTeam", {}).get("nickName", ""))
        away = _std_team(d.get("awayTeam", {}).get("nickName", ""))

        officials = d.get("officials", [])
        referee = ""
        for o in officials:
            if o.get("position") == "Referee":
                referee = f"{o.get('firstName', '')} {o.get('lastName', '')}".strip()
                break

        rows.append({
            "year": year,
            "round": str(rnd),
            "home_team": home,
            "away_team": away,
            "referee": referee,
            "ground_conditions": d.get("groundConditions", ""),
            "weather": d.get("weather", ""),
        })
        time.sleep(delay)

    return rows


def backfill_officials(start_year=2013, end_year=2025, delay=0.2) -> pd.DataFrame:
    """Scrape all officials data and save to parquet."""
    all_rows = []
    for year in range(start_year, end_year + 1):
        cache = CACHE_DIR / f"officials_{year}.json"
        if cache.exists():
            with open(cache) as f:
                rows = json.load(f)
            print(f"  {year}: {len(rows)} matches (cached)")
            all_rows.extend(rows)
            continue

        print(f"  {year}: scraping...")
        year_rows = []
        empty_streak = 0
        for rnd in range(1, 28):
            rows = fetch_round_officials(year, rnd, delay=delay)
            if rows:
                year_rows.extend(rows)
                empty_streak = 0
            else:
                empty_streak += 1
                if empty_streak >= 2:
                    break
            time.sleep(delay)

        print(f"    → {len(year_rows)} matches")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(year_rows, f)
        all_rows.extend(year_rows)
        time.sleep(0.5)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    OFFICIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OFFICIALS_PATH, index=False)
    print(f"\n  Saved {len(df)} rows to {OFFICIALS_PATH}")

    # Summary
    refs = df[df["referee"] != ""]
    print(f"  Refs found: {refs['referee'].nunique()} unique across {len(refs)} games")

    return df


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2013)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--delay", type=float, default=0.15)
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("  NRL Officials Scraper")
    print("=" * 60)
    backfill_officials(args.start_year, args.end_year, args.delay)


if __name__ == "__main__":
    main()
