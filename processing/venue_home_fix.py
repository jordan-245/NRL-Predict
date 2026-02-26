"""
Fix home/away team assignment using venue data.

RLP round summaries list the winning team first, not the home team.
Without odds data for cross-referencing, we use venue → team mappings
to determine the actual home team.

Usage:
    from processing.venue_home_fix import fix_home_away
    matches = fix_home_away(matches_df)
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# ============================================================
# Venue → Home Team mapping (historical venue names included)
# ============================================================
# Only includes unambiguous home-ground venues.
# Neutral venues (e.g. Vegas, Adelaide Oval) are excluded.

VENUE_HOME_TEAM: dict[str, str] = {
    # Brisbane Broncos
    "Suncorp Stadium": "Brisbane Broncos",
    "Lang Park": "Brisbane Broncos",
    "Brisbane Cricket Ground": "Brisbane Broncos",

    # Melbourne Storm
    "AAMI Park": "Melbourne Storm",
    "Melbourne Rectangular Stadium": "Melbourne Storm",
    "Olympic Park": "Melbourne Storm",

    # Canberra Raiders
    "GIO Stadium": "Canberra Raiders",
    "Canberra Stadium": "Canberra Raiders",
    "Centrebet Stadium": "Canberra Raiders",
    "Bruce Stadium": "Canberra Raiders",

    # Sydney Roosters
    "Allianz Stadium": "Sydney Roosters",
    "Sydney Cricket Ground": "Sydney Roosters",
    "Sydney Football Stadium": "Sydney Roosters",

    # Gold Coast Titans
    "Cbus Super Stadium": "Gold Coast Titans",
    "Robina Stadium": "Gold Coast Titans",
    "Skilled Park": "Gold Coast Titans",

    # Newcastle Knights
    "McDonald Jones Stadium": "Newcastle Knights",
    "Hunter Stadium": "Newcastle Knights",
    "EnergyAustralia Stadium": "Newcastle Knights",

    # North Queensland Cowboys
    "1300SMILES Stadium": "North Queensland Cowboys",
    "Queensland Country Bank Stadium": "North Queensland Cowboys",
    "Dairy Farmers Stadium": "North Queensland Cowboys",

    # New Zealand Warriors
    "Mt Smart Stadium": "New Zealand Warriors",
    "Go Media Stadium": "New Zealand Warriors",

    # Parramatta Eels
    "CommBank Stadium": "Parramatta Eels",
    "Bankwest Stadium": "Parramatta Eels",
    "Pirtek Stadium": "Parramatta Eels",
    "Parramatta Stadium": "Parramatta Eels",

    # Manly Sea Eagles
    "4 Pines Park": "Manly Sea Eagles",
    "Brookvale Oval": "Manly Sea Eagles",
    "Lottoland": "Manly Sea Eagles",

    # St George Illawarra Dragons
    "WIN Stadium": "St George Illawarra Dragons",
    "Netstrata Jubilee Stadium": "St George Illawarra Dragons",
    "Jubilee Stadium": "St George Illawarra Dragons",
    "WIN Jubilee Stadium": "St George Illawarra Dragons",
    "UOW Jubilee Oval": "St George Illawarra Dragons",

    # Cronulla Sharks
    "Pointsbet Stadium": "Cronulla Sharks",
    "Southern Cross Group Stadium": "Cronulla Sharks",
    "Remondis Stadium": "Cronulla Sharks",
    "Sharks Stadium": "Cronulla Sharks",
    "Shark Stadium": "Cronulla Sharks",
    "Shark Park": "Cronulla Sharks",
    "Toyota Stadium": "Cronulla Sharks",
    "Kayo Stadium": "Cronulla Sharks",
    "Ocean Protect Stadium": "Cronulla Sharks",

    # Penrith Panthers
    "BlueBet Stadium": "Penrith Panthers",
    "Panthers Stadium": "Penrith Panthers",
    "Pepper Stadium": "Penrith Panthers",
    "Penrith Stadium": "Penrith Panthers",
    "Sportingbet Stadium": "Penrith Panthers",

    # Wests Tigers (two primary grounds)
    "Campbelltown Sports Stadium": "Wests Tigers",
    "Leichhardt Oval": "Wests Tigers",

    # South Sydney Rabbitohs
    "Accor Stadium": "South Sydney Rabbitohs",
    "Stadium Australia": "South Sydney Rabbitohs",

    # Canterbury Bulldogs
    "Belmore Sports Ground": "Canterbury Bulldogs",

    # Dolphins
    "Sunshine Coast Stadium": "Dolphins",
    "Moreton Daily Stadium": "Dolphins",

    # Shared / multi-team venues - assigned to primary tenant for the era
    # ANZ Stadium was Canterbury's primary home 2013-2019, then demolished
    "ANZ Stadium": "Canterbury Bulldogs",

    # Central Coast - used by several teams as secondary home
    "Central Coast Stadium": "Newcastle Knights",
    "Bluetongue Stadium": "Newcastle Knights",

    # BB Print Stadium (Mackay) - Cowboys regional
    "BB Print Stadium": "North Queensland Cowboys",
    "Barlow Park": "North Queensland Cowboys",

    # TIO Stadium (Darwin) - used by multiple teams on rotation
    # Cowboys most frequent
    "TIO Stadium": "North Queensland Cowboys",
}


def fix_home_away(df: pd.DataFrame) -> pd.DataFrame:
    """Fix home/away assignment using venue data.

    For each match, checks if the venue belongs to one of the two teams.
    If the venue team is currently in the away_team column, swaps both
    teams and their associated scores/stats.

    Returns a new DataFrame with corrected home/away columns.
    """
    df = df.copy()

    swap_cols_pairs = [
        ("home_team", "away_team"),
        ("home_score", "away_score"),
    ]
    # Also swap halftime and penalty columns if present
    for prefix in ["halftime", "penalty"]:
        hcol = f"{prefix}_home"
        acol = f"{prefix}_away"
        if hcol in df.columns and acol in df.columns:
            swap_cols_pairs.append((hcol, acol))

    swapped = 0
    kept = 0
    neutral = 0

    for idx, row in df.iterrows():
        venue = row.get("venue")
        if pd.isna(venue):
            neutral += 1
            continue

        venue_team = VENUE_HOME_TEAM.get(venue)
        if venue_team is None:
            neutral += 1
            continue

        home = row["home_team"]
        away = row["away_team"]

        if venue_team == home:
            # Already correct
            kept += 1
        elif venue_team == away:
            # Need to swap
            for hcol, acol in swap_cols_pairs:
                df.at[idx, hcol], df.at[idx, acol] = row[acol], row[hcol]
            swapped += 1
        else:
            # Venue team doesn't match either team — neutral venue
            neutral += 1

    total = len(df)
    scored = df["home_score"].notna()
    home_wins = (df.loc[scored, "home_score"] > df.loc[scored, "away_score"]).sum()
    away_wins = (df.loc[scored, "home_score"] < df.loc[scored, "away_score"]).sum()
    draws = (df.loc[scored, "home_score"] == df.loc[scored, "away_score"]).sum()
    home_pct = home_wins / (home_wins + away_wins) * 100 if (home_wins + away_wins) > 0 else 0

    print(f"  Home/away fix: {swapped} swapped, {kept} kept, {neutral} neutral/unmatched")
    print(f"  Result: {home_wins} home wins, {away_wins} away wins, {draws} draws ({home_pct:.1f}% home win rate)")

    return df
