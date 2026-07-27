"""
sim_v23_A3c_convergence_plots.py -- Phase 3 Series A3c: Convergence Curve Visualization

Series A3c (Session 5, sim_v22_A3c_seedcheck.py) found the D=175 Pareto claim
fails seed robustness (temp/paper1-A3c-seed-robustness-findings.md). This
script reruns the identical configurations -- same seeds, same parameters,
fully deterministic -- purely to capture the per-timestep convergence
history (mean_cosim_hist) that sim_v22 computed internally but never saved,
so the *shape* of the seed divergence can be seen, not just final summary
numbers.

**Not a new experiment.** Adds seed=42 (the original A3a/A3b baseline) to
the 5 A3c seeds so all 6 seeds -- 42, 11, 99, 7, 3, 21 -- run at each of
D=150/175/200 (18 runs total) in one consistent plotting pass, instead of
mixing this rerun with the original separate A3b/highD-sweep plots.

Fixed parameters -- identical to sim_v22_A3c_seedcheck.py / sim_v21_A3b.py:
  N=30, M=3, ALPHA=0.3, ETA=0.15, LAMBDA=0.1, KAPPA=0.0
  WS k=4 p=0.1 topology, Gaussian init, T=5000

Determinism is verified, not assumed: this script asserts the rerun's 15
A3c final eq_cosim values match A3c-summary.csv exactly, and the rerun's 3
seed=42 final eq_cosim values match the existing A3b-summary.csv /
highD-sweep-summary.csv baseline exactly (both within float tolerance) --
any mismatch is flagged as a bug, not silently reconciled.

Outputs (workspace/simulation/phase-3/):
  sim_v23_A3c_convergence_plots.png  -- 1x3 panels (D=150/175/200), 6 seed
                                         curves per panel, shared y-axis
  A3c-convergence-history.csv        -- columns: D, seed, t, mean_cosim

Run from project root:
  python workspace/simulation/phase-3/sim_v23_A3c_convergence_plots.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
import os

# -- Fixed parameters (identical to sim_v22_A3c_seedcheck.py) -----------------

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
SEEDS    = [42, 11, 99, 7, 3, 21]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
A3B_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "A3b-fine-D-sweep")
A3A_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "A3a-highD-sweep")

SEED_COLORS = {
    42: "#000000",
    11: "#2196F3",
    99: "#4CAF50",
    7:  "#FF5722",
    3:  "#9C27B0",
    21: "#FF9800",
}

# -- Helper functions (unchanged from sim_v22_A3c_seedcheck.py) --------------

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

def run_one(D, seed):
    print()
    print("=" * 72)
    print(f"D={D}  seed={seed}")
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
    mean_over_topics = mean_cosim_hist.mean(axis=1)   # per-timestep, averaged over M topics
    t_to_eq_arr = np.where(mean_over_topics >= 0.95 * eq_cosim)[0]
    t_to_eq = int(t_to_eq_arr[0]) if len(t_to_eq_arr) > 0 else T

    minSME = min(ever_sme)
    maxHolder = max(top_holder)

    print(f"--- D={D} seed={seed} summary --- eq_cosim={eq_cosim:.4f}  t_to_eq={t_to_eq}  "
          f"minSME={minSME}/{N}  maxHolder={maxHolder:.4f}")

    return {
        "D": D, "seed": seed,
        "eq_cosim": eq_cosim, "t_to_eq": t_to_eq,
        "minSME": minSME, "maxHolder": maxHolder,
        "mean_over_topics": mean_over_topics,   # (T+1,) history -- the new capture
    }


# -- Main ----------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print("sim_v23_A3c_convergence_plots.py -- Phase 3 Series A3c: convergence curves")
print(f"D values: {D_VALUES}  |  Seeds: {SEEDS}  |  18 runs (deterministic rerun)")
print(f"Fixed: N={N} M={M} ALPHA={ALPHA} ETA={ETA} LAMBDA={LAMBDA} KAPPA={KAPPA} T={T}")
print("=" * 72)

results = []
for D in D_VALUES:
    for seed in SEEDS:
        res = run_one(D, seed)
        results.append(res)

# -- Console summary table -----------------------------------------------------

print()
print("=" * 72)
print("SUMMARY TABLE -- Series A3c rerun (6 seeds x 3 D values, incl. seed=42)")
print(f"{'D':>4}  {'Seed':>5}  {'eq_cosim':>9}  {'t_to_eq':>8}  "
      f"{'minSME':>7}  {'maxHolder':>10}")
print("-" * 72)
for r in results:
    print(f"{r['D']:>4}  {r['seed']:>5}  {r['eq_cosim']:>9.4f}  "
          f"{r['t_to_eq']:>8}  {r['minSME']:>7}  {r['maxHolder']:>10.4f}")

# -- Write raw per-timestep history CSV ----------------------------------------

history_csv_path = os.path.join(OUTPUT_DIR, "A3c-convergence-history.csv")
with open(history_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["D", "seed", "t", "mean_cosim"])
    for r in results:
        D = r["D"]
        seed = r["seed"]
        for t, mc in enumerate(r["mean_over_topics"]):
            writer.writerow([D, seed, t, f"{mc:.6f}"])
print(f"\nHistory CSV saved -> {history_csv_path}")

# -- Plot: 1x3 panels, 6 seed curves per panel, shared y-axis ------------------

all_vals = np.concatenate([r["mean_over_topics"] for r in results])
y_lo = float(np.min(all_vals)) - 0.03
y_hi = float(np.max(all_vals)) + 0.03

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle(
    "sim_v23_A3c_convergence_plots.py -- Series A3c convergence curves | "
    f"N={N} M={M} ALPHA={ALPHA} ETA={ETA} KAPPA={KAPPA} T={T}\n"
    "6 seeds per panel (42 = original A3a/A3b baseline, 11/99/7/3/21 = A3c new seeds)",
    fontsize=9
)

for i, D in enumerate(D_VALUES):
    ax = axes[i]
    for seed in SEEDS:
        r = next(res for res in results if res["D"] == D and res["seed"] == seed)
        label = f"seed={seed}" + (" (baseline)" if seed == 42 else "")
        ax.plot(r["mean_over_topics"], color=SEED_COLORS[seed], linewidth=1.0,
                alpha=0.85, label=label)
    ax.set_title(f"D={D}", fontsize=10)
    ax.set_xlabel("Timestep", fontsize=8)
    if i == 0:
        ax.set_ylabel("Mean pairwise cosim (avg over 3 topics)", fontsize=8)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "sim_v23_A3c_convergence_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved -> {plot_path}")

print()
print("-- sim_v23_A3c_convergence_plots.py complete --")


# =============================================================================
# --- Test: rerun reproduces Session 5 values ---
# =============================================================================
# A3c-summary.csv stores eq_cosim rounded to 4 decimals. Since this rerun is
# bit-for-bit deterministic (same seed drives both the graph topology and the
# rng), the freshly-computed eq_cosim rounds to the identical 4-decimal value
# -- so the tolerance below is checking for exact reproduction, not proximity.

A3C_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "A3c-summary.csv")
with open(A3C_SUMMARY_PATH, newline="") as f:
    _a3c_rows = list(csv.DictReader(f))

_a3c_recorded = {
    (int(row["D"]), int(row["seed"])): float(row["eq_cosim"])
    for row in _a3c_rows
}

_A3C_SEEDS = [11, 99, 7, 3, 21]
for D in D_VALUES:
    for seed in _A3C_SEEDS:
        rerun_val = round(
            next(r for r in results if r["D"] == D and r["seed"] == seed)["eq_cosim"],
            4
        )
        recorded_val = _a3c_recorded[(D, seed)]
        assert abs(rerun_val - recorded_val) < 1e-9, (
            f"D={D} seed={seed}: rerun eq_cosim={rerun_val} does not match "
            f"A3c-summary.csv recorded value={recorded_val}"
        )

# =============================================================================
# --- Test: seed=42 rerun matches the existing baseline ---
# =============================================================================

A3B_SUMMARY_PATH = os.path.join(A3B_DIR, "A3b-summary.csv")
with open(A3B_SUMMARY_PATH, newline="") as f:
    _a3b_rows = list(csv.DictReader(f))

_a3b_baseline = {
    int(row["D"]): float(row["eq_cosim"])
    for row in _a3b_rows
    if row["subrun"] == "D-sweep" and row["ETA"] == "0.15" and int(row["D"]) in (150, 175)
}

HIGHD_SUMMARY_PATH = os.path.join(A3A_DIR, "highD-sweep-summary.csv")
with open(HIGHD_SUMMARY_PATH, newline="") as f:
    _highd_rows = list(csv.DictReader(f))

_highd_baseline = {int(row["D"]): float(row["eq_cosim"]) for row in _highd_rows}

_seed42_baseline = {150: _a3b_baseline[150], 175: _a3b_baseline[175], 200: _highd_baseline[200]}

for D in D_VALUES:
    rerun_val = round(
        next(r for r in results if r["D"] == D and r["seed"] == 42)["eq_cosim"], 4
    )
    baseline_val = _seed42_baseline[D]
    assert abs(rerun_val - baseline_val) < 1e-9, (
        f"D={D} seed=42: rerun eq_cosim={rerun_val} does not match the existing "
        f"baseline value={baseline_val} (A3b-summary.csv / highD-sweep-summary.csv) "
        f"-- this is a real discrepancy, not silently reconciled"
    )

print("[PASS] Test: rerun determinism confirmed against Session 5 and the original baseline")
