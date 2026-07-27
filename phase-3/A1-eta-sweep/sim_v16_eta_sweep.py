"""
sim_v16_eta_sweep.py -- Phase 3 Series A1: ETA Sweep at D=5

Series A1 runs (Runs 33-50): 6 ETA values x 3 seeds = 18 runs.
Finds the ETA threshold where the diversity regime becomes stable
across seeds (no longer determined by initial geometry alone).

Context: Series A0 showed ETA=0.15 at D=5 is a critical zone --
1/9 seeds in full diversity, cosim std 0.35-0.38. Seeds chosen:
  seed=21  (diversity end of A0: all topics < 0.5)
  seed=42  (reference: T1~0.8, T2/T3~0.1-0.2)
  seed=123 (partial convergence: T2~0.9, T1/T3~0.2-0.6)

Fixed: N=30, M=3, D=5, T=500, ALPHA=0.3, LAMBDA=0.1, KAPPA=0.0
Variable: ETA in {0.01, 0.05, 0.07, 0.10, 0.12, 0.15}

Run numbering:
  ETA=0.01: Runs 33-35 (seeds 21, 42, 123)
  ETA=0.05: Runs 36-38
  ETA=0.07: Runs 39-41
  ETA=0.10: Runs 42-44
  ETA=0.12: Runs 45-47
  ETA=0.15: Runs 48-50  (reference -- should reproduce A0 for these 3 seeds)

Outputs (workspace/simulation/phase-3/):
  sim_v16_vector_plots.png  -- 6x3 grid (rows=ETA, cols=seed)
  eta-sweep-summary.csv     -- one row per (ETA, seed)

Run from project root:
  python workspace/simulation/phase-3/sim_v16_eta_sweep.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
import os

# -- Fixed parameters ---------------------------------------------------------

N       = 30
M       = 3
D       = 5
T       = 500
ALPHA   = 0.3
LAMBDA  = 0.1
KAPPA   = 0.0
CONV_THRESHOLD = 0.99
SME_PCT = 90

ETA_VALUES = [0.01, 0.05, 0.07, 0.10, 0.12, 0.15]
SEEDS      = [21, 42, 123]

TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]
TOPIC_LABELS = [f"Topic {k + 1}" for k in range(M)]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Helper functions (unchanged from sim_v14/v15) ----------------------------

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


# -- Single-run simulation (ETA is now a parameter) --------------------------

def run_one(run_num, seed, eta):
    """
    Run one (ETA, seed) combination. Returns result dict.
    """
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
            b_raw = (1 - ALPHA - eta) * B[:, k, :] + ALPHA * social + eta * innovation
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

    sme_fraction = SME_hist.mean(axis=0)
    ever_sme     = [int(np.any(SME_hist[:, :, k], axis=0).sum()) for k in range(M)]
    top_holder   = [float(sme_fraction[:, k].max()) for k in range(M)]
    final_cosims = [float(mean_cosim_hist[-1, k]) for k in range(M)]

    diversity_regime = all(c < 0.5 for c in final_cosims)
    converged_flag   = any(c > CONV_THRESHOLD for c in final_cosims)

    conv_all   = np.all(mean_cosim_hist > CONV_THRESHOLD, axis=1)
    conv_steps = np.where(conv_all)[0]
    t_conv = int(conv_steps[0]) if len(conv_steps) > 0 else None

    return {
        "run_num":          run_num,
        "eta":              eta,
        "seed":             seed,
        "mean_cosim_hist":  mean_cosim_hist,
        "final_cosims":     final_cosims,
        "ever_sme":         ever_sme,
        "top_holder":       top_holder,
        "diversity_regime": diversity_regime,
        "converged":        converged_flag,
        "t_conv":           t_conv,
    }


# -- Run all (ETA, seed) combinations ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print("sim_v16_eta_sweep.py -- Phase 3 Series A1 | ETA sweep D=5 | Runs 33-50")
print(f"N={N}  M={M}  D={D}  T={T}  ALPHA={ALPHA}  LAMBDA={LAMBDA}  KAPPA={KAPPA}")
print(f"ETA values: {ETA_VALUES}")
print(f"Seeds: {SEEDS}")
print("=" * 72)

results = []
run_num = 33
for eta in ETA_VALUES:
    for seed in SEEDS:
        print(f"  Run {run_num:02d}  ETA={eta:.2f}  seed={seed:>3} ...", end=" ", flush=True)
        r = run_one(run_num, seed, eta)
        results.append(r)
        cosims_str = "  ".join(f"T{k+1}:{r['final_cosims'][k]:.3f}" for k in range(M))
        print(f"done |  {cosims_str}  |  "
              f"minSME={min(r['ever_sme'])}/{N}  "
              f"div={'Y' if r['diversity_regime'] else 'N'}  "
              f"conv={'Y' if r['converged'] else 'N'}")
        run_num += 1


# -- Console summary table ---------------------------------------------------

print(f"\n{'=' * 72}")
print(f"SUMMARY TABLE -- Series A1  D={D}  T={T}  N={N}  ALPHA={ALPHA}")
print(f"{'Run':>4}  {'ETA':>5}  {'Seed':>5}  {'T1':>7}  {'T2':>7}  {'T3':>7}  "
      f"{'minSME':>7}  {'maxTop':>7}  {'div':>4}  {'conv':>5}")
print("-" * 72)
for r in results:
    print(f"{r['run_num']:>4}  {r['eta']:>5.2f}  {r['seed']:>5}  "
          f"{r['final_cosims'][0]:>7.3f}  "
          f"{r['final_cosims'][1]:>7.3f}  "
          f"{r['final_cosims'][2]:>7.3f}  "
          f"{min(r['ever_sme']):>7d}  "
          f"{max(r['top_holder']):>7.3f}  "
          f"{'Y' if r['diversity_regime'] else 'N':>4}  "
          f"{'Y' if r['converged'] else 'N':>5}")

# Per-ETA summary: how many of 3 seeds in diversity / converged
print(f"\nPer-ETA summary (div = all topics cosim < 0.5, conv = any topic cosim > {CONV_THRESHOLD}):")
print(f"{'ETA':>5}  {'div/3':>6}  {'conv/3':>7}  {'mean T1':>9}  {'mean T2':>9}  {'mean T3':>9}")
print("-" * 55)
for eta in ETA_VALUES:
    eta_results = [r for r in results if r['eta'] == eta]
    n_div  = sum(1 for r in eta_results if r['diversity_regime'])
    n_conv = sum(1 for r in eta_results if r['converged'])
    mean_cosims = [
        np.mean([r['final_cosims'][k] for r in eta_results]) for k in range(M)
    ]
    print(f"{eta:>5.2f}  {n_div:>6d}  {n_conv:>7d}  "
          f"{mean_cosims[0]:>9.3f}  {mean_cosims[1]:>9.3f}  {mean_cosims[2]:>9.3f}")


# -- Write CSV ---------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "eta-sweep-summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run_num", "eta", "seed",
        "topic1_cosim_T500", "topic2_cosim_T500", "topic3_cosim_T500",
        "ever_sme_t1", "ever_sme_t2", "ever_sme_t3",
        "min_ever_sme", "max_top_holder",
        "diversity_regime", "converged", "t_conv",
    ])
    for r in results:
        writer.writerow([
            r["run_num"],
            f"{r['eta']:.2f}",
            r["seed"],
            f"{r['final_cosims'][0]:.4f}",
            f"{r['final_cosims'][1]:.4f}",
            f"{r['final_cosims'][2]:.4f}",
            r["ever_sme"][0], r["ever_sme"][1], r["ever_sme"][2],
            min(r["ever_sme"]),
            f"{max(r['top_holder']):.4f}",
            int(r["diversity_regime"]),
            int(r["converged"]),
            r["t_conv"] if r["t_conv"] is not None else "NA",
        ])
print(f"\nCSV saved -> {csv_path}")


# -- Plots: 6x3 grid (rows=ETA, cols=seed) ------------------------------------

fig, axes = plt.subplots(6, 3, figsize=(15, 18))
fig.suptitle(
    f"sim_v16_eta_sweep.py -- Phase 3 Series A1 | ETA sweep | D={D}, N={N}, T={T}\n"
    f"ALPHA={ALPHA}, KAPPA={KAPPA}, Gaussian init | "
    f"Dashed: conv threshold ({CONV_THRESHOLD}) | Dotted: 0.5 diversity boundary",
    fontsize=10
)

for row_i, eta in enumerate(ETA_VALUES):
    for col_j, seed in enumerate(SEEDS):
        r = next(x for x in results if x['eta'] == eta and x['seed'] == seed)
        ax = axes[row_i][col_j]

        for k in range(M):
            ax.plot(r["mean_cosim_hist"][:, k],
                    color=TOPIC_COLORS[k], linewidth=0.9,
                    label=TOPIC_LABELS[k] if (row_i == 0 and col_j == 0) else None)
        ax.axhline(CONV_THRESHOLD, color="black", linestyle="--",
                   linewidth=0.7, alpha=0.8)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_title(
            f"Run {r['run_num']}  ETA={eta:.2f}  seed={seed}  "
            f"div={'Y' if r['diversity_regime'] else 'N'}  "
            f"conv={'Y' if r['converged'] else 'N'}",
            fontsize=8
        )
        ax.set_xlabel("Timestep", fontsize=7)
        ax.set_ylabel("Mean cosim", fontsize=7)
        ax.set_ylim(-0.3, 1.05)
        ax.tick_params(labelsize=7)

axes[0][0].legend(fontsize=7, loc="lower right")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "sim_v16_vector_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plots saved -> {plot_path}")
print("\n-- sim_v16_eta_sweep.py complete --")
