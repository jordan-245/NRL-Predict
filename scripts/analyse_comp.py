#!/usr/bin/env python3
"""Analyse 2025 tipping comp results to find edge."""

import numpy as np

# 2025 results parsed from the leaderboard
comp = {
    "Krob": {
        "rounds": [6,4,7,3,3,7,3,5,4,4,4,1,4,7,3,4,8,4,6,5,5,5,7,7,4,6,6],
        "finals": [2,1,1,1], "gf_margin": 2, "total": 137, "margin": 360, "rank": 1
    },
    "ESPNFAN": {
        "rounds": [6,4,6,2,4,5,4,5,2,3,4,1,7,6,4,4,7,3,6,8,5,5,5,5,5,7,7],
        "finals": [3,2,1,0], "gf_margin": 12, "total": 136, "margin": 417, "rank": 2
    },
    "Jordan": {
        "rounds": [6,4,6,5,3,5,5,5,4,5,5,2,6,6,4,5,6,2,6,5,6,6,6,4,3,6,6],
        "finals": [2,1,1,0], "gf_margin": 8, "total": 136, "margin": 419, "rank": 3
    },
    "Henry": {
        "rounds": [6,4,6,2,4,4,5,5,4,5,5,0,6,6,4,5,7,3,6,6,6,5,6,4,3,6,5],
        "finals": [4,2,2,0], "gf_margin": 36, "total": 136, "margin": 467, "rank": 4
    },
    "Salesy": {
        "rounds": [5,3,7,5,5,5,4,6,4,3,4,0,5,7,3,3,7,4,6,5,5,4,5,7,6,6,6],
        "finals": [2,1,1,1], "gf_margin": 6, "total": 135, "margin": 463, "rank": 5
    },
    "Payne": {
        "rounds": [7,4,7,2,5,3,3,4,4,2,4,0,5,6,3,5,8,3,7,7,6,4,5,5,5,5,5],
        "finals": [2,1,1,1], "gf_margin": 2, "total": 129, "margin": 465, "rank": 6
    },
}

print("=" * 70)
print("  2025 TIPPING COMP ANALYSIS — Finding Jordan's Edge")
print("=" * 70)

# 1. How close was the race?
print("\n── THE RACE ──")
print(f"  1st Krob:    137 tips (margin 360)")
print(f"  2nd ESPNFAN: 136 tips (margin 417)")
print(f"  3rd Jordan:  136 tips (margin 419)")
print(f"")
print(f"  Jordan was 1 TIP from winning.")
print(f"  Jordan was 2 MARGIN POINTS from 2nd.")

# 2. Regular season vs finals
print("\n── REGULAR SEASON vs FINALS ──")
for name, d in comp.items():
    reg = sum(d["rounds"])
    fin = sum(d["finals"])
    print(f"  {name:<12s}: reg {reg:>3d}/216   finals {fin:>2d}/9   "
          f"total {d['total']:>3d}   margin {d['margin']:>3d}")

# 3. Per-round comparison: Jordan vs Krob
print("\n── ROUND-BY-ROUND: Jordan vs Krob ──")
j = comp["Jordan"]["rounds"]
k = comp["Krob"]["rounds"]

jordan_better = 0
krob_better = 0
tied = 0
biggest_gap_j = (0, 0)
biggest_gap_k = (0, 0)

print(f"  {'Rnd':>4s} {'Jordan':>7s} {'Krob':>7s} {'Δ':>5s}")
for i in range(27):
    diff = j[i] - k[i]
    marker = ""
    if diff > 0:
        jordan_better += 1
        marker = f"  Jordan +{diff}"
        if diff > biggest_gap_j[1]:
            biggest_gap_j = (i+1, diff)
    elif diff < 0:
        krob_better += 1
        marker = f"  Krob +{-diff}"
        if -diff > biggest_gap_k[1]:
            biggest_gap_k = (i+1, -diff)
    else:
        tied += 1
    print(f"  R{i+1:>2d}  {j[i]:>5d}   {k[i]:>5d}  {diff:>+3d}{marker}")

print(f"\n  Jordan won {jordan_better} rounds, Krob won {krob_better}, tied {tied}")
print(f"  Jordan's best: R{biggest_gap_j[0]} (+{biggest_gap_j[1]})")
print(f"  Krob's best:   R{biggest_gap_k[0]} (+{biggest_gap_k[1]})")

# Regular season totals
j_reg = sum(j)
k_reg = sum(k)
print(f"\n  Regular season: Jordan {j_reg}, Krob {k_reg} (Δ{j_reg-k_reg:+d})")

# Finals
j_fin = comp["Jordan"]["finals"]
k_fin = comp["Krob"]["finals"]
print(f"  Finals: Jordan {sum(j_fin)} (QF:{j_fin[0]} SF:{j_fin[1]} PF:{j_fin[2]} GF:{j_fin[3]})")
print(f"          Krob   {sum(k_fin)} (QF:{k_fin[0]} SF:{k_fin[1]} PF:{k_fin[2]} GF:{k_fin[3]})")
print(f"  Finals gap: {sum(j_fin)-sum(k_fin):+d}")

# 4. Consistency analysis
print("\n── CONSISTENCY ──")
for name in ["Jordan", "Krob", "ESPNFAN", "Henry"]:
    r = comp[name]["rounds"]
    avg = np.mean(r)
    std = np.std(r)
    low_rounds = sum(1 for x in r if x <= 3)
    high_rounds = sum(1 for x in r if x >= 6)
    print(f"  {name:<12s}: avg {avg:.1f}/8  std {std:.1f}  "
          f"bad(≤3) {low_rounds:>2d}  great(≥6) {high_rounds:>2d}")

# 5. The REAL analysis: where do tips differ?
print("\n── WHERE DOES JORDAN LOSE GROUND? ──")
for opp_name in ["Krob"]:
    opp = comp[opp_name]["rounds"]
    lost_rounds = []
    won_rounds = []
    for i in range(27):
        diff = j[i] - opp[i]
        if diff < 0:
            lost_rounds.append((i+1, j[i], opp[i], diff))
        elif diff > 0:
            won_rounds.append((i+1, j[i], opp[i], diff))

    print(f"\n  Rounds Jordan LOST to {opp_name}:")
    total_lost = 0
    for rnd, jv, ov, d in lost_rounds:
        total_lost += abs(d)
        print(f"    R{rnd:>2d}: Jordan {jv}, {opp_name} {ov} ({d:+d})")
    print(f"    Total ground lost: {total_lost} tips")

    print(f"\n  Rounds Jordan WON vs {opp_name}:")
    total_won = 0
    for rnd, jv, ov, d in won_rounds:
        total_won += abs(d)
        print(f"    R{rnd:>2d}: Jordan {jv}, {opp_name} {ov} ({d:+d})")
    print(f"    Total ground won: {total_won} tips")

# 6. Field analysis - what's the realistic ceiling?
print("\n── FIELD ANALYSIS ──")
# Best possible score per round (if you tipped perfectly among the field)
max_per_round = []
for i in range(27):
    best = max(d["rounds"][i] for d in comp.values())
    max_per_round.append(best)

print(f"  Best available per round: {max_per_round}")
print(f"  Sum of best-per-round: {sum(max_per_round)} (impossible, but ceiling)")
print(f"  Average best-per-round: {np.mean(max_per_round):.1f}/8")

# 7. Margin analysis
print("\n── MARGIN TIEBREAKER ──")
print(f"  Jordan's total margin error: 419")
print(f"  Krob's total margin error:   360")
print(f"  ESPNFAN's margin error:      417")
print(f"  Jordan margin error / round: {419/27:.1f}")
print(f"  Krob margin error / round:   {360/27:.1f}")
print(f"")
print(f"  If Jordan matched Krob's tips (137), Jordan needs:")
print(f"    margin < 360 to win on tiebreaker")
print(f"  If Jordan ties on tips (137), need avg margin error < {360/27:.1f} per round")

# 8. Strategy implications
print("\n── STRATEGY IMPLICATIONS ──")
print("""
  1. THE GAP IS TINY: 1 tip over 27 rounds + finals.
     Getting 1 extra tip right wins the comp.

  2. JORDAN'S WEAKNESS: Rounds 5,17,25 (3 or 2/8).
     These are the "disaster rounds" — everyone has them,
     but minimising bad rounds matters more than maximising
     good ones.

  3. MARGIN MATTERS: Jordan lost 2nd place by 2 margin
     points. Better margin predictions = free ranking boost
     when tips are tied.

  4. FINALS ARE DECISIVE: Jordan went 4/9, Krob went 5/9.
     That 1-tip gap came entirely from the Grand Final.
     Getting GF right = winning the comp.

  5. CONSISTENCY > HEROICS: Jordan's consistency (std 1.2)
     is already good. The win comes from avoiding the
     2-3/8 rounds, not chasing 8/8 rounds.
""")


if __name__ == "__main__":
    main = None
    # runs at import
