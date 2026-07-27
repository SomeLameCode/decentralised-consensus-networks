"""
degree-verification.py -- Series A3c: minimum-degree verification per seed

Verifies the claim made in Paper 1 Section 4.9 about seed 11's outlier
status: that its network has one node at unusually low degree (2),
versus every other tested seed's minimum of 3.

This was not computed or saved anywhere in sim_v22_A3c_seedcheck.py --
the original script only used each seed's degree sequence internally
(for the isolated-node NaN-guard) without ever printing or persisting
the minimum degree. This script regenerates the exact same graphs
(identical topology parameters and seed) purely to make that claim
independently checkable, without re-running the simulation itself.

Seeds: the six used across Series A3c plus the seed=42 baseline from
Series A3b, i.e. every seed contributing to the eq_cosim ≈ 0.76-0.83
range discussed in Section 4.9.

Run from project root:
    python workspace/simulation/phase-3/A3c-seed-robustness/degree-verification.py
"""

import networkx as nx
import csv
import os

N = 30
K = 4
P = 0.1

SEEDS = [11, 99, 7, 3, 21, 42]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []
for seed in SEEDS:
    G = nx.watts_strogatz_graph(N, k=K, p=P, seed=seed)
    degrees = sorted(d for _, d in G.degree())
    results.append({
        "seed": seed,
        "min_degree": degrees[0],
        "degree_sequence": degrees,
    })

print("=" * 60)
print("Series A3c -- minimum-degree verification per seed")
print(f"N={N} k={K} p={P} (Watts-Strogatz)")
print("=" * 60)
for r in results:
    flag = "  <-- outlier" if r["min_degree"] < 3 else ""
    print(f"seed={r['seed']:>3}  min_degree={r['min_degree']}{flag}")

csv_path = os.path.join(OUTPUT_DIR, "degree-verification.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "min_degree", "degree_sequence"])
    for r in results:
        writer.writerow([r["seed"], r["min_degree"], r["degree_sequence"]])
print(f"\nCSV saved -> {csv_path}")

# -- Test: confirms the Paper 1 Section 4.9 claim exactly --
seed11_min = next(r["min_degree"] for r in results if r["seed"] == 11)
others_min = [r["min_degree"] for r in results if r["seed"] != 11]

assert seed11_min == 2, f"expected seed 11 min_degree=2, got {seed11_min}"
assert all(m == 3 for m in others_min), f"expected all other seeds min_degree=3, got {others_min}"

print("\n[PASS] Confirms Paper 1 Section 4.9: seed 11 has min_degree=2, "
      "all other seeds (99, 7, 3, 21, 42) have min_degree=3.")
