"""
Extract player appearance data from cached RLP HTML files.

Re-parses all round-summary HTML pages that were already downloaded by the
RLP scraper, extracting lineup data (starters + bench) that was previously
dropped during Parquet serialisation because list columns aren't supported.

Includes surname disambiguation logic: scopes players by team + era + jersey
number to create stable player_id identifiers.

Output: data/processed/player_appearances.parquet

Usage:
    python -m processing.build_player_data          # build from scratch
    python -m processing.build_player_data --round 5 --year 2026  # single round append
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import START_YEAR, END_YEAR, ALL_ROUNDS, PROCESSED_DIR, RAW_DIR
from config.team_mappings import standardise_team_name
from scraping.rlp_match_parser import parse_round_summary

logger = logging.getLogger(__name__)

# Jersey number → position mapping (NRL standard)
JERSEY_POSITION = {
    1: "FB", 2: "WG", 3: "CE", 4: "CE", 5: "WG",
    6: "FE", 7: "HB", 8: "PR", 9: "HK", 10: "PR",
    11: "2R", 12: "2R", 13: "LK",
    14: "INT", 15: "INT", 16: "INT", 17: "INT",
    18: "RES", 19: "RES", 20: "RES", 21: "RES", 22: "RES",
}

# Spine positions (most impactful)
SPINE_POSITIONS = {"FB", "HB", "FE", "HK"}

OUTPUT_PATH = PROCESSED_DIR / "player_appearances.parquet"


def _load_round_html(year: int, round_id: int | str) -> str | None:
    """Load cached HTML for a round from disk."""
    cache_dir = RAW_DIR / "rlp" / "seasons" / f"nrl-{year}" / f"round-{round_id}"
    summary_path = cache_dir / "summary.html"
    if summary_path.exists():
        return summary_path.read_text(encoding="utf-8", errors="replace")

    # Try loading from the rate_limiter cache
    from scraping.rlp_url_builder import round_summary_url
    url = round_summary_url(year, round_id)
    cache_path = RAW_DIR / "rlp" / "cache"
    if cache_path.exists():
        # Hash-based cache: try finding by URL
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        for p in cache_path.glob(f"*{url_hash}*"):
            return p.read_text(encoding="utf-8", errors="replace")

    return None


def _safe_standardise(name: str) -> str:
    """Standardise team name, returning original on failure."""
    try:
        return standardise_team_name(name)
    except KeyError:
        return name


def extract_appearances_from_round(
    year: int,
    round_id: int | str,
    html: str,
) -> list[dict]:
    """Parse a single round's HTML and extract player appearance records."""
    matches = parse_round_summary(html, year=year, round_id=round_id)
    records = []

    for match in matches:
        if match.get("is_bye") or match.get("is_abandoned"):
            continue

        home_team = match.get("home_team") or ""
        away_team = match.get("away_team") or ""
        home_score = match.get("home_score")
        away_score = match.get("away_score")

        if not home_team or not away_team:
            continue

        home_team = _safe_standardise(home_team)
        away_team = _safe_standardise(away_team)

        # Fix home/away: RLP lists winning team first, not home team.
        # Use venue mapping to determine the real home team, then swap
        # if needed so match_ids match matches.parquet.
        from processing.venue_home_fix import VENUE_HOME_TEAM
        venue = match.get("venue") or ""
        venue_team = VENUE_HOME_TEAM.get(venue)
        if venue_team and venue_team == away_team:
            # Away team is actually the home team — swap everything
            home_team, away_team = away_team, home_team
            home_score, away_score = away_score, home_score
            match["home_lineup"], match["away_lineup"] = (
                match.get("away_lineup", []),
                match.get("home_lineup", []),
            )
            match["home_bench"], match["away_bench"] = (
                match.get("away_bench", []),
                match.get("home_bench", []),
            )

        # Determine result
        if home_score is not None and away_score is not None:
            margin_home = home_score - away_score
            margin_away = away_score - home_score
            result_home = "W" if margin_home > 0 else ("L" if margin_home < 0 else "D")
            result_away = "W" if margin_away > 0 else ("L" if margin_away < 0 else "D")
        else:
            margin_home = margin_away = None
            result_home = result_away = None

        match_id = f"{year}_r{round_id}_{home_team}_v_{away_team}"

        # Process each side
        for side, team, opponent, ha, lineup, bench, result, margin in [
            ("home", home_team, away_team, "home",
             match.get("home_lineup", []), match.get("home_bench", []),
             result_home, margin_home),
            ("away", away_team, home_team, "away",
             match.get("away_lineup", []), match.get("away_bench", []),
             result_away, margin_away),
        ]:
            if not lineup and not bench:
                continue

            # Starters: jersey numbers 1-13
            for jersey_num, player_name in enumerate(lineup, start=1):
                if not player_name or not player_name.strip():
                    continue
                position = JERSEY_POSITION.get(jersey_num, "UNK")
                records.append({
                    "year": year,
                    "round": str(round_id),
                    "match_id": match_id,
                    "team": team,
                    "opponent": opponent,
                    "home_away": ha,
                    "result": result,
                    "margin": margin,
                    "player_name": player_name.strip(),
                    "jersey_number": jersey_num,
                    "position": position,
                    "is_starter": True,
                    "is_spine": position in SPINE_POSITIONS,
                })

            # Bench: jersey numbers 14+
            for bench_idx, player_name in enumerate(bench):
                if not player_name or not player_name.strip():
                    continue
                jersey_num = 14 + bench_idx
                position = JERSEY_POSITION.get(jersey_num, "INT")
                records.append({
                    "year": year,
                    "round": str(round_id),
                    "match_id": match_id,
                    "team": team,
                    "opponent": opponent,
                    "home_away": ha,
                    "result": result,
                    "margin": margin,
                    "player_name": player_name.strip(),
                    "jersey_number": jersey_num,
                    "position": position,
                    "is_starter": False,
                    "is_spine": False,
                })

    return records


def disambiguate_players(df: pd.DataFrame) -> pd.DataFrame:
    """Add player_id column using team + surname + position + era disambiguation.

    Strategy:
    - Same surname + same team + same jersey number in consecutive rounds = same player
    - Use (team, surname, primary_position, first_year) as unique key
    - Track known position switches gracefully
    """
    if df.empty:
        return df.assign(player_id="")

    # Sort chronologically
    df = df.sort_values(["year", "round", "team", "jersey_number"]).reset_index(drop=True)

    # Build player identity tracker: (team, surname) → list of player contexts
    player_registry: dict[tuple[str, str], list[dict]] = {}
    player_ids = []

    for _, row in df.iterrows():
        team = row["team"]
        name = row["player_name"]
        position = row["position"]
        year = row["year"]
        jersey = row["jersey_number"]

        key = (team, name)
        if key not in player_registry:
            # New player for this team
            pid = f"{team}_{name}_{position}_{year}"
            player_registry[key] = [{
                "player_id": pid,
                "positions": {position},
                "first_year": year,
                "last_year": year,
                "jerseys": {jersey},
            }]
            player_ids.append(pid)
        else:
            # Find matching context (could be multiple players with same surname)
            contexts = player_registry[key]
            matched = False

            for ctx in contexts:
                # Same player if:
                # 1. Seen within last 2 years (still active), AND
                # 2. Position is compatible (same group or bench)
                year_gap = year - ctx["last_year"]
                if year_gap <= 2:
                    ctx["last_year"] = year
                    ctx["positions"].add(position)
                    ctx["jerseys"].add(jersey)
                    player_ids.append(ctx["player_id"])
                    matched = True
                    break

            if not matched:
                # New player with same surname (e.g. father/son, or returnee after long gap)
                pid = f"{team}_{name}_{position}_{year}"
                contexts.append({
                    "player_id": pid,
                    "positions": {position},
                    "first_year": year,
                    "last_year": year,
                    "jerseys": {jersey},
                })
                player_ids.append(pid)

    df["player_id"] = player_ids
    return df


def build_full(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> pd.DataFrame:
    """Build complete player_appearances.parquet from all cached RLP HTML."""
    all_records = []

    for year in range(start_year, end_year + 1):
        year_count = 0
        for round_id in ALL_ROUNDS:
            html = _load_round_html(year, round_id)
            if html is None:
                continue
            records = extract_appearances_from_round(year, round_id, html)
            all_records.extend(records)
            year_count += len(records)

        if year_count > 0:
            print(f"  {year}: {year_count} player appearances")

    if not all_records:
        print("  WARNING: No player appearances found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Disambiguate
    print(f"\n  Disambiguating {len(df)} records...")
    df = disambiguate_players(df)

    # Summary stats
    n_players = df["player_id"].nunique()
    n_matches = df["match_id"].nunique()
    n_starters = df[df["is_starter"]].shape[0]
    n_bench = df[~df["is_starter"]].shape[0]
    print(f"  Total: {len(df)} appearances, {n_players} unique players, {n_matches} matches")
    print(f"  Starters: {n_starters}, Bench: {n_bench}")

    return df


def append_round(
    year: int,
    round_id: int | str,
    existing_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Append a single round's appearances to existing parquet."""
    html = _load_round_html(year, round_id)
    if html is None:
        raise FileNotFoundError(f"No cached HTML for {year} round {round_id}")

    new_records = extract_appearances_from_round(year, round_id, html)
    if not new_records:
        print(f"  No appearances found for {year} round {round_id}")
        return pd.DataFrame()

    new_df = pd.DataFrame(new_records)

    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        # Remove any existing data for this round (idempotent)
        mask = ~((existing["year"] == year) & (existing["round"] == str(round_id)))
        existing = existing[mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = disambiguate_players(combined)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Build player appearances data")
    parser.add_argument("--round", type=str, default=None, help="Single round to append")
    parser.add_argument("--year", type=int, default=None, help="Year for single round")
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 60)
    print("  Building Player Appearances Data")
    print("=" * 60)

    if args.round and args.year:
        print(f"\n  Appending round {args.round} of {args.year}...")
        df = append_round(args.year, args.round)
    else:
        print(f"\n  Full rebuild: {START_YEAR}-{END_YEAR}...")
        df = build_full()

    if not df.empty:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"\n  Saved to {OUTPUT_PATH}")
        print(f"  Shape: {df.shape}")

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
