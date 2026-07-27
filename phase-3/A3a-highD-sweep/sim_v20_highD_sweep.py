"""
sim_v20_highD_sweep.py -- Phase 3 Series A3a: High-D sweep to find Pareto equilibrium

Context: D=20 at ETA=0.15 produces equilibrium cosim ~0.977 -- too much consensus.
Looking for D where equilibrium sits at cosim ~0.80 (Pareto point: strong collective
direction with meaningful residual diversity). Noise amplitude = sigma*sqrt(D) so
higher D -> more angular perturbation per step -> lower equilibrium cosim.

Fixed: ALPHA=0.3, ETA=0.15, KAPPA=0.0, Gaussian init, WS k=4 p=0.1, seed=42, M=3, N=30
Variable: D in {20, 50, 100, 200, 500}
T=5000 for all. Milestone every 500 steps.
D=20 first as anchor validation (should reproduce cosim ~0.977).

Outputs (workspace/simulation/phase-3/):
  sim_v20_highD_sweep_plots.png  (1x5 grid, one panel per D)
  highD-sweep-summary.csv

Run from project root:
  python workspace/simulation/phase-3/sim_v20_highD_sweep.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
import os
import sys

# -- Parameters ---------------------------------------------------------------

N       = 30
M       = 3
ALPHA   = 0.3
ETA     = 0.15
LAMBDA  = 0.1
KAPPA   = 0.0
SEED    = 42
T       = 5000

D_VALUES      = [20, 50, 100, 200, 500]
D_TO_RUN      = {20: 60, 50: 61, 100: 62, 200: 63, 500: 64}
ANCHOR_D      = 20
ANCHOR_EXPECTED = 0.977
ANCHOR_TOL    = 0.02

CONV_THRESHOLD = 0.99
PARETO_TARGET  = 0.80
SME_PCT        = 90
MILESTONE_STEP = 500
EQ_START       = 1000

SNAP_TIMES     = [100, 500, 1000, 5000]

TOPIC_COLORS   = ["#2196F3", "#4CAF50", "#FF5722"]
TOPIC_LABELS   = [f"T{k + 1}" for k in range(M)]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Helper functions ---------------------------------------------------------

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


# -- Single D run -------------------------------------------------------------

def run_one_D(D):
    run_num = D_TO_RUN[D]

    print()
    print("=" * 72)
    print(f"Run {run_num} -- D={D} | seed={SEED} | T={T}")
    print(f"N={N}  M={M}  ALPHA={ALPHA}  ETA={ETA}  KAPPA={KAPPA}")
    print(f"Milestone every {MILESTONE_STEP} steps | eq window t={EQ_START}-{T}")
    print("=" * 72)

    rng = np.random.default_rng(SEED)
    G = nx.watts_strogatz_graph(N, k=4, p=0.1, seed=SEED)
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

    print(f"Initial cosim: " +
          "  ".join(f"T{k+1}:{mean_cosim_hist[0, k]:.3f}" for k in range(M)))
    print()

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

        if (t + 1) % MILESTONE_STEP == 0:
            cosims_str = "  ".join(
                f"T{k+1}:{mean_cosim_hist[t+1, k]:.3f}" for k in range(M)
            )
            mean_all = float(np.mean(mean_cosim_hist[t + 1, :]))
            print(f"  t={t+1:>5}  |  {cosims_str}  |  mean={mean_all:.3f}")

    # -- Derived metrics --------------------------------------------------

    sme_fraction = SME_hist.mean(axis=0)
    ever_sme     = [int(np.any(SME_hist[:, :, k], axis=0).sum()) for k in range(M)]
    top_holder   = [float(sme_fraction[:, k].max()) for k in range(M)]

    eq_cosim = float(np.mean(mean_cosim_hist[EQ_START:, :]))

    mean_over_topics = mean_cosim_hist.mean(axis=1)
    eq_thresh_95 = 0.95 * eq_cosim
    t_to_eq_arr = np.where(mean_over_topics >= eq_thresh_95)[0]
    t_to_eq = int(t_to_eq_arr[0]) if len(t_to_eq_arr) > 0 else T

    snapshots = {}
    for snap_t in SNAP_TIMES:
        if snap_t <= T:
            snapshots[snap_t] = [float(mean_cosim_hist[snap_t, k]) for k in range(M)]

    print()
    print(f"--- D={D} summary ---")
    print(f"eq_cosim (t={EQ_START}-{T}, all topics): {eq_cosim:.4f}  "
          f"(target={PARETO_TARGET}, delta={abs(eq_cosim - PARETO_TARGET):.4f})")
    print(f"t_to_eq (95% of eq): t={t_to_eq}")
    print(f"Snapshots per topic:")
    for snap_t in SNAP_TIMES:
        vals = "  ".join(f"T{k+1}:{snapshots[snap_t][k]:.3f}" for k in range(M))
        mean_s = float(np.mean(snapshots[snap_t]))
        print(f"  t={snap_t:>5}: {vals}  mean={mean_s:.3f}")
    print(f"ever-SME: {ever_sme}  (min={min(ever_sme)}/{N})")
    print(f"top_holder: {[round(x, 3) for x in top_holder]}")

    return {
        "D": D,
        "run": run_num,
        "eq_cosim": eq_cosim,
        "t_to_eq": t_to_eq,
        "ever_sme": ever_sme,
        "top_holder": top_holder,
        "snapshots": snapshots,
        "mean_cosim_hist": mean_cosim_hist,
    }


# -- Main ---------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print("sim_v20_highD_sweep.py -- Phase 3 Series A3a")
print(f"High-D sweep: D in {D_VALUES}")
print(f"Fixed: N={N}  M={M}  ALPHA={ALPHA}  ETA={ETA}  KAPPA={KAPPA}  "
      f"seed={SEED}  T={T}")
print(f"Pareto target: eq_cosim ~ {PARETO_TARGET}")
print(f"Anchor: D={ANCHOR_D} should reproduce eq_cosim ~ {ANCHOR_EXPECTED} "
      f"(tol +/- {ANCHOR_TOL})")
print("=" * 72)

results = []

for D in D_VALUES:
    res = run_one_D(D)
    results.append(res)

    if D == ANCHOR_D:
        if abs(res["eq_cosim"] - ANCHOR_EXPECTED) > ANCHOR_TOL:
            print()
            print(f"ANCHOR VALIDATION FAILED for D={ANCHOR_D}")
            print(f"  eq_cosim = {res['eq_cosim']:.4f}  "
                  f"expected {ANCHOR_EXPECTED} +/- {ANCHOR_TOL}")
            print("Stopping. Check rng initialisation or T budget.")
            sys.exit(1)
        else:
            print(f"Anchor D={ANCHOR_D} VALIDATED: eq_cosim={res['eq_cosim']:.4f} "
                  f"(expected ~{ANCHOR_EXPECTED})")

# -- Summary table ------------------------------------------------------------

print()
print("=" * 72)
print("SUMMARY TABLE -- Series A3a High-D sweep")
print(f"{'D':>5}  {'Run':>4}  {'eq_cosim':>9}  {'t_to_eq':>8}  "
      f"{'minSME':>7}  {'maxHolder':>10}  {'delta_target':>12}")
print("-" * 72)

closest_d = None
closest_delta = float("inf")
for res in results:
    delta = abs(res["eq_cosim"] - PARETO_TARGET)
    if delta < closest_delta:
        closest_delta = delta
        closest_d = res["D"]

for res in results:
    d = res["D"]
    delta = abs(res["eq_cosim"] - PARETO_TARGET)
    marker = " <-- closest to Pareto" if d == closest_d else ""
    print(f"{d:>5}  {res['run']:>4}  {res['eq_cosim']:>9.4f}  "
          f"{res['t_to_eq']:>8}  {min(res['ever_sme']):>7}  "
          f"{max(res['top_holder']):>10.3f}  {delta:>12.4f}{marker}")

print()
closest_res = next(r for r in results if r["D"] == closest_d)
print(f"Closest D to Pareto target (cosim={PARETO_TARGET}): "
      f"D={closest_d}  eq_cosim={closest_res['eq_cosim']:.4f}")

# -- Write CSV ----------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "highD-sweep-summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["D", "run", "eq_cosim", "t_to_eq", "delta_pareto"]
    for snap_t in SNAP_TIMES:
        for k in range(M):
            header.append(f"snap_t{snap_t}_T{k+1}")
    header += ["ever_sme_T1", "ever_sme_T2", "ever_sme_T3",
               "min_ever_sme", "max_top_holder"]
    writer.writerow(header)

    for res in results:
        row = [
            res["D"], res["run"],
            f"{res['eq_cosim']:.4f}", res["t_to_eq"],
            f"{abs(res['eq_cosim'] - PARETO_TARGET):.4f}",
        ]
        for snap_t in SNAP_TIMES:
            for k in range(M):
                row.append(f"{res['snapshots'][snap_t][k]:.4f}")
        row += [
            res["ever_sme"][0], res["ever_sme"][1], res["ever_sme"][2],
            min(res["ever_sme"]),
            f"{max(res['top_holder']):.4f}",
        ]
        writer.writerow(row)

print(f"\nCSV saved -> {csv_path}")

# -- Plot: 1x5 grid -----------------------------------------------------------

fig, axes = plt.subplots(1, len(D_VALUES), figsize=(22, 4), sharey=False)
fig.suptitle(
    "sim_v20_highD_sweep.py -- Series A3a High-D sweep | "
    f"seed={SEED} | T={T} | N={N}, ALPHA={ALPHA}, ETA={ETA}\n"
    "Orange line = eq_cosim | Purple dashed = Pareto target (0.80) | "
    "Black dashed = conv threshold (0.99) | Grey dotted = eq window start",
    fontsize=8
)

for i, res in enumerate(results):
    ax = axes[i]
    D = res["D"]
    hist = res["mean_cosim_hist"]

    for k in range(M):
        ax.plot(hist[:, k], color=TOPIC_COLORS[k], linewidth=0.7,
                label=TOPIC_LABELS[k], alpha=0.9)

    ax.axhline(CONV_THRESHOLD, color="black", linestyle="--",
               linewidth=0.8, alpha=0.6, label=f"conv({CONV_THRESHOLD})")
    ax.axhline(PARETO_TARGET, color="purple", linestyle="--",
               linewidth=0.9, alpha=0.8, label=f"target({PARETO_TARGET})")
    ax.axhline(res["eq_cosim"], color="orange", linestyle="-",
               linewidth=1.0, alpha=0.9, label=f"eq={res['eq_cosim']:.3f}")
    ax.axvline(EQ_START, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)

    delta = abs(res["eq_cosim"] - PARETO_TARGET)
    marker_str = " [*]" if res["D"] == closest_d else ""
    ax.set_title(
        f"D={D}  Run {res['run']}\neq={res['eq_cosim']:.3f}  "
        f"dt={delta:.3f}{marker_str}",
        fontsize=8
    )
    ax.set_xlabel("Timestep", fontsize=7)
    if i == 0:
        ax.set_ylabel("Mean pairwise cosim", fontsize=7)
    ax.set_ylim(-0.15, 1.05)
    ax.legend(fontsize=6, loc="lower right")
    ax.tick_params(labelsize=6)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "sim_v20_highD_sweep_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plots saved -> {plot_path}")
print()
print("-- sim_v20_highD_sweep.py complete --")
