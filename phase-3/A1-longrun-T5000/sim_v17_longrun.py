"""
sim_v17_longrun.py -- Phase 3 Series A1 extended: T=5000 long run

Motivation: A1 ETA sweep (Runs 33-50) showed zero convergence at T=500
across all ETA values 0.01-0.15. Lower ETA gives MORE inertia (higher
self-weight), producing LOWER cosim at T=500, not convergence. The question
is whether the system converges at all given sufficient time, or whether the
diversity pattern is truly structural in D=5.

This run fixes ETA=0.15 (Run 14 / A0 reference baseline), extends to T=5000,
and runs the 3 A1 seeds to observe long-run trajectory.

Seeds: {21, 42, 123}  (diversity / reference / partial from A0 and A1)
Fixed: N=30, M=3, D=5, ALPHA=0.3, ETA=0.15, LAMBDA=0.1, KAPPA=0.0
       Gaussian init, WS k=4 p=0.1

Run numbering: Runs 51-53

Outputs (workspace/simulation/phase-3/):
  sim_v17_longrun_plots.png  -- 1x3 grid, one panel per seed
  longrun-summary.csv

Run from project root:
  python workspace/simulation/phase-3/sim_v17_longrun.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
import os

# -- Parameters ---------------------------------------------------------------

N       = 30
M       = 3
D       = 5
T       = 5000
ALPHA   = 0.3
ETA     = 0.15
LAMBDA  = 0.1
KAPPA   = 0.0
CONV_THRESHOLD = 0.99
SME_PCT = 90

SEEDS = [21, 42, 123]

TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]
TOPIC_LABELS = [f"Topic {k + 1}" for k in range(M)]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Helper functions (unchanged from sim_v14/v15/v16) ------------------------

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


# -- Single-seed long run -----------------------------------------------------

def run_one(run_num, seed):
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

    # Progress milestones
    milestones = {500, 1000, 2000, 3000, 4000, 5000}

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

        if (t + 1) in milestones:
            cosims_str = "  ".join(
                f"T{k+1}:{mean_cosim_hist[t+1, k]:.3f}" for k in range(M)
            )
            print(f"    t={t+1:>5}  |  {cosims_str}")

    sme_fraction = SME_hist.mean(axis=0)
    ever_sme     = [int(np.any(SME_hist[:, :, k], axis=0).sum()) for k in range(M)]
    top_holder   = [float(sme_fraction[:, k].max()) for k in range(M)]
    final_cosims = [float(mean_cosim_hist[-1, k]) for k in range(M)]

    diversity_regime = all(c < 0.5 for c in final_cosims)
    converged_flag   = any(c > CONV_THRESHOLD for c in final_cosims)

    conv_all   = np.all(mean_cosim_hist > CONV_THRESHOLD, axis=1)
    conv_steps = np.where(conv_all)[0]
    t_conv = int(conv_steps[0]) if len(conv_steps) > 0 else None

    # Snapshot cosims at key timepoints
    snapshots = {}
    for snap_t in [500, 1000, 2000, 5000]:
        snapshots[snap_t] = [float(mean_cosim_hist[snap_t, k]) for k in range(M)]

    return {
        "run_num":          run_num,
        "seed":             seed,
        "mean_cosim_hist":  mean_cosim_hist,
        "final_cosims":     final_cosims,
        "snapshots":        snapshots,
        "ever_sme":         ever_sme,
        "top_holder":       top_holder,
        "diversity_regime": diversity_regime,
        "converged":        converged_flag,
        "t_conv":           t_conv,
    }


# -- Main ---------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print("sim_v17_longrun.py -- Phase 3 Series A1 extended | T=5000 | Runs 51-53")
print(f"N={N}  M={M}  D={D}  T={T}  ALPHA={ALPHA}  ETA={ETA}  KAPPA={KAPPA}")
print(f"Seeds: {SEEDS}")
print("=" * 72)

results = []
for i, seed in enumerate(SEEDS):
    run_num = 51 + i
    print(f"\n  Run {run_num:02d}  ETA={ETA:.2f}  seed={seed:>3}  T={T}")
    r = run_one(run_num, seed)
    results.append(r)
    cosims_str = "  ".join(f"T{k+1}:{r['final_cosims'][k]:.3f}" for k in range(M))
    print(f"  --> done | final cosim: {cosims_str} | "
          f"div={'Y' if r['diversity_regime'] else 'N'}  "
          f"conv={'Y' if r['converged'] else 'N'}")


# -- Console summary ----------------------------------------------------------

print(f"\n{'=' * 72}")
print(f"SUMMARY TABLE -- Series A1 extended  T={T}  ETA={ETA}  D={D}")
print(f"{'Run':>4}  {'Seed':>5}  {'T1 final':>9}  {'T2 final':>9}  {'T3 final':>9}  "
      f"{'minSME':>7}  {'div':>4}  {'conv':>5}")
print("-" * 72)
for r in results:
    print(f"{r['run_num']:>4}  {r['seed']:>5}  "
          f"{r['final_cosims'][0]:>9.3f}  "
          f"{r['final_cosims'][1]:>9.3f}  "
          f"{r['final_cosims'][2]:>9.3f}  "
          f"{min(r['ever_sme']):>7d}  "
          f"{'Y' if r['diversity_regime'] else 'N':>4}  "
          f"{'Y' if r['converged'] else 'N':>5}")

print(f"\nCosim snapshots across time:")
print(f"{'':>12}  {'t=500':>9}  {'t=1000':>9}  {'t=2000':>9}  {'t=5000':>9}")
for r in results:
    for k in range(M):
        label = f"s{r['seed']} {TOPIC_LABELS[k]}"
        vals = [f"{r['snapshots'][snap_t][k]:>9.3f}"
                for snap_t in [500, 1000, 2000, 5000]]
        print(f"  {label:<10}  {'  '.join(vals)}")
    print()


# -- Write CSV ----------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "longrun-summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run_num", "seed",
        "T1_t500", "T2_t500", "T3_t500",
        "T1_t1000", "T2_t1000", "T3_t1000",
        "T1_t2000", "T2_t2000", "T3_t2000",
        "T1_t5000", "T2_t5000", "T3_t5000",
        "min_ever_sme", "max_top_holder",
        "diversity_regime", "converged", "t_conv",
    ])
    for r in results:
        row = [r["run_num"], r["seed"]]
        for snap_t in [500, 1000, 2000, 5000]:
            for k in range(M):
                row.append(f"{r['snapshots'][snap_t][k]:.4f}")
        row += [
            min(r["ever_sme"]),
            f"{max(r['top_holder']):.4f}",
            int(r["diversity_regime"]),
            int(r["converged"]),
            r["t_conv"] if r["t_conv"] is not None else "NA",
        ]
        writer.writerow(row)
print(f"CSV saved -> {csv_path}")


# -- Plots: 1x3 grid, one panel per seed -------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"sim_v17_longrun.py -- Series A1 extended | T={T}  ETA={ETA}  D={D}  N={N}\n"
    f"ALPHA={ALPHA}, KAPPA={KAPPA}, Gaussian init | "
    f"Dashed: conv threshold ({CONV_THRESHOLD}) | Dotted: 0.5 diversity boundary",
    fontsize=10
)

for j, r in enumerate(results):
    ax = axes[j]
    for k in range(M):
        ax.plot(r["mean_cosim_hist"][:, k],
                color=TOPIC_COLORS[k], linewidth=0.8, label=TOPIC_LABELS[k])
    ax.axhline(CONV_THRESHOLD, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.7)
    # Milestone markers
    for snap_t in [500, 1000, 2000]:
        ax.axvline(snap_t, color="lightgrey", linestyle="-", linewidth=0.5, alpha=0.6)
    ax.set_title(
        f"Run {r['run_num']}  seed={r['seed']}  "
        f"div={'Y' if r['diversity_regime'] else 'N'}  "
        f"conv={'Y' if r['converged'] else 'N'}",
        fontsize=9
    )
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("Mean pairwise cosim", fontsize=8)
    ax.set_ylim(-0.15, 1.05)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "sim_v17_longrun_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plots saved -> {plot_path}")
print("\n-- sim_v17_longrun.py complete --")
