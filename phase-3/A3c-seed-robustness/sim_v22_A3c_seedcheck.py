"""
sim_v22_A3c_seedcheck.py -- Phase 3 Series A3c: Pareto Point Seed-Robustness Check

Tests whether the D=175 Pareto equilibrium claim -- seed=42 baseline:
  D=150 eq_cosim=0.8221 (Run 66, A3b-summary.csv)
  D=175 eq_cosim=0.7954, t_to_eq=79 (Run 67, A3b-summary.csv)
  D=200 eq_cosim=0.7688, t_to_eq=90 (Run 63, highD-sweep-summary.csv)
-- holds across new seeds, not just seed=42.

Runs 70-84: D in {150, 175, 200} x 5 new seeds (11, 99, 7, 3, 21) = 15 runs.

Fixed parameters -- identical to sim_v21_A3b.py's sub-run 1 (only the seed,
for both graph topology and rng, changes):
  N=30, M=3, ALPHA=0.3, ETA=0.15, LAMBDA=0.1, KAPPA=0.0
  WS k=4 p=0.1 topology, Gaussian init, T=5000

Existing seed=42 data (A3b-summary.csv, highD-sweep-summary.csv) is the
untouched baseline -- not re-run or modified here. Combining these 15 new
rows with that baseline into a 6-seed-per-D picture and applying the 4
success criteria is Prompt 5b's job, not this script's.

Outputs (workspace/simulation/phase-3/):
  A3c-summary.csv   -- columns: D, seed, eq_cosim, t_to_eq, minSME, maxHolder

Run from project root:
  python workspace/simulation/phase-3/sim_v22_A3c_seedcheck.py
"""

import numpy as np
import networkx as nx
import csv
import os

# -- Fixed parameters (identical to sim_v21_A3b.py's sub-run 1) --------------

N       = 30
M       = 3
ALPHA   = 0.3
ETA     = 0.15
LAMBDA  = 0.1
KAPPA   = 0.0
T       = 5000

EQ_START = 1000
SME_PCT  = 90

D_VALUES = [150, 175, 200]
SEEDS    = [11, 99, 7, 3, 21]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

RUN_NUMBERS = {}
_run = 70
for _D in D_VALUES:
    for _seed in SEEDS:
        RUN_NUMBERS[(_D, _seed)] = _run
        _run += 1

# -- Helper functions (unchanged from sim_v21_A3b.py) -------------------------

def compute_trust(A, c_k):
    W = A * c_k[np.newaxis, :]
    row_sums = W.sum(axis=1, keepdims=True)
    A_row_sums = A.sum(axis=1, keepdims=True)
    fallback = A / np.where(A_row_sums == 0, 1, A_row_sums)
    return np.where(row_sums > 0, W / np.where(row_sums > 0, row_sums, 1), fallback)


def apply_bounded_confidence_cosine(W, B_k, kappa):
    cosim = B_k @ B_k.T
    mask = (cosim >= kappa).astype(float)
    np.fill_diagonal(mask, 0.0)
    W_bc = W * mask
    row_sums = W_bc.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    return np.where(row_sums > 0, W_bc / safe_sums, 0.0)


def minmax_unit_nan_safe(x):
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi > lo:
        return (x - lo) / (hi - lo)
    return np.where(np.isnan(x), np.nan, 0.5)


def mean_pairwise_cosim(B_k):
    cosim = B_k @ B_k.T
    idx = np.triu_indices(N, k=1)
    return float(cosim[idx].mean())


# -- Single run (one D x seed configuration) ----------------------------------

def run_one(D, seed, run_num):
    print()
    print("=" * 72)
    print(f"Run {run_num} -- D={D}  seed={seed}")
    print(f"N={N}  M={M}  ALPHA={ALPHA}  ETA={ETA}  KAPPA={KAPPA}  T={T}")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    G = nx.watts_strogatz_graph(N, k=4, p=0.1, seed=seed)
    A = nx.to_numpy_array(G)
    degrees = np.array([d for _, d in G.degree()])
    isolated = degrees == 0

    B_raw = rng.standard_normal((N, M, D))
    norms = np.linalg.norm(B_raw, axis=2, keepdims=True)
    B = B_raw / np.where(norms > 0, norms, 1.0)
    C = rng.uniform(0.1, 0.9, (N, M))

    mean_cosim_hist = np.zeros((T + 1, M))
    SME_hist = np.zeros((T + 1, N, M), dtype=bool)

    for k in range(M):
        mean_cosim_hist[0, k] = mean_pairwise_cosim(B[:, k, :])
        threshold = np.percentile(C[:, k], SME_PCT)
        SME_hist[0, :, k] = C[:, k] >= threshold

    for t in range(T):
        B_new = np.empty_like(B)
        C_new = np.empty_like(C)

        for k in range(M):
            W_k = compute_trust(A, C[:, k])
            W_k = apply_bounded_confidence_cosine(W_k, B[:, k, :], KAPPA)

            social = W_k @ B[:, k, :]
            innovation = rng.normal(0.0, 0.1, (N, D))
            b_raw = (1 - ALPHA - ETA) * B[:, k, :] + ALPHA * social + ETA * innovation
            norms_new = np.linalg.norm(b_raw, axis=1, keepdims=True)
            B_new[:, k, :] = b_raw / np.where(norms_new > 0, norms_new, 1.0)

            cosim_matrix = B[:, k, :] @ B[:, k, :].T
            delta = (W_k * cosim_matrix).sum(axis=1)
            delta[isolated] = np.nan
            delta_scaled = minmax_unit_nan_safe(delta)
            delta_scaled[isolated] = 0.0

            C_new[:, k] = np.clip(
                (1 - LAMBDA) * C[:, k] + LAMBDA * delta_scaled,
                0.0, 1.0
            )

        B, C = B_new, C_new

        for k in range(M):
            mean_cosim_hist[t + 1, k] = mean_pairwise_cosim(B[:, k, :])
            threshold = np.percentile(C[:, k], SME_PCT)
            SME_hist[t + 1, :, k] = C[:, k] >= threshold

        if (t + 1) % 1000 == 0:
            mean_all = float(np.mean(mean_cosim_hist[t + 1, :]))
            print(f"  t={t + 1:>5}  mean_cosim={mean_all:.3f}")

    # -- Derived metrics --------------------------------------------------------

    sme_fraction = SME_hist.mean(axis=0)
    ever_sme   = [int(np.any(SME_hist[:, :, k], axis=0).sum()) for k in range(M)]
    top_holder = [float(sme_fraction[:, k].max()) for k in range(M)]

    eq_cosim = float(np.mean(mean_cosim_hist[EQ_START:, :]))
    mean_over_topics = mean_cosim_hist.mean(axis=1)
    t_to_eq_arr = np.where(mean_over_topics >= 0.95 * eq_cosim)[0]
    t_to_eq = int(t_to_eq_arr[0]) if len(t_to_eq_arr) > 0 else T

    minSME = min(ever_sme)
    maxHolder = max(top_holder)

    print(f"--- Run {run_num} summary --- eq_cosim={eq_cosim:.4f}  t_to_eq={t_to_eq}  "
          f"minSME={minSME}/{N}  maxHolder={maxHolder:.4f}")

    return {
        "D": D, "seed": seed, "run": run_num,
        "eq_cosim": eq_cosim, "t_to_eq": t_to_eq,
        "minSME": minSME, "maxHolder": maxHolder,
    }


# -- Main ----------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print("sim_v22_A3c_seedcheck.py -- Phase 3 Series A3c: Pareto seed-robustness check")
print(f"D values: {D_VALUES}  |  Seeds: {SEEDS}  |  15 runs (70-84)")
print(f"Fixed: N={N} M={M} ALPHA={ALPHA} ETA={ETA} LAMBDA={LAMBDA} KAPPA={KAPPA} T={T}")
print("=" * 72)

results = []
for D in D_VALUES:
    for seed in SEEDS:
        run_num = RUN_NUMBERS[(D, seed)]
        res = run_one(D, seed, run_num)
        results.append(res)

# -- Console summary table -----------------------------------------------------

print()
print("=" * 72)
print("SUMMARY TABLE -- Series A3c (5 new seeds x 3 D values; seed=42 baseline")
print("already on record in A3b-summary.csv / highD-sweep-summary.csv, not repeated here)")
print(f"{'Run':>4}  {'D':>4}  {'Seed':>5}  {'eq_cosim':>9}  {'t_to_eq':>8}  "
      f"{'minSME':>7}  {'maxHolder':>10}")
print("-" * 72)
for r in results:
    print(f"{r['run']:>4}  {r['D']:>4}  {r['seed']:>5}  {r['eq_cosim']:>9.4f}  "
          f"{r['t_to_eq']:>8}  {r['minSME']:>7}  {r['maxHolder']:>10.4f}")

# -- Write CSV ------------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "A3c-summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["D", "seed", "eq_cosim", "t_to_eq", "minSME", "maxHolder"])
    for r in results:
        writer.writerow([
            r["D"], r["seed"],
            f"{r['eq_cosim']:.4f}", r["t_to_eq"],
            r["minSME"], f"{r['maxHolder']:.4f}",
        ])
print(f"\nCSV saved -> {csv_path}")

print()
print("-- sim_v22_A3c_seedcheck.py complete --")


# =============================================================================
# --- Test: A3c run completeness ---
# =============================================================================

EXPECTED_PAIRS = {(D, seed) for D in D_VALUES for seed in SEEDS}

with open(csv_path, newline="") as f:
    _reader = csv.DictReader(f)
    _rows = list(_reader)

assert len(_rows) == 15, f"expected 15 rows in A3c-summary.csv, found {len(_rows)}"

_seen_pairs = [(int(row["D"]), int(row["seed"])) for row in _rows]
assert len(_seen_pairs) == len(set(_seen_pairs)), \
    "duplicate (D, seed) pairs found in A3c-summary.csv"
assert set(_seen_pairs) == EXPECTED_PAIRS, (
    f"(D, seed) pairs mismatch -- expected {sorted(EXPECTED_PAIRS)}, "
    f"got {sorted(_seen_pairs)}"
)

# --- Test: A3c fixed parameters ---

A3B_PARAMS = {
    "N": 30, "M": 3, "ALPHA": 0.3, "ETA": 0.15, "LAMBDA": 0.1, "KAPPA": 0.0,
    "T": 5000, "ws_k": 4, "ws_p": 0.1, "init": "gaussian",
}
A3C_PARAMS = {
    "N": N, "M": M, "ALPHA": ALPHA, "ETA": ETA, "LAMBDA": LAMBDA, "KAPPA": KAPPA,
    "T": T, "ws_k": 4, "ws_p": 0.1, "init": "gaussian",
}
assert A3C_PARAMS == A3B_PARAMS, (
    f"A3c fixed parameters diverge from sim_v21_A3b.py -- "
    f"A3c={A3C_PARAMS}  A3b={A3B_PARAMS}"
)

print("[PASS] Test: A3c run completeness and parameter fidelity")
